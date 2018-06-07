
# coding: utf-8

# In[230]:


from IPython.display import display
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import math
import numpy as np
import pandas as pd
from scipy import signal,interpolate
import os
import shutil

import wfdb


# In[231]:


# 定义求RR间期的函数
def RR(signal,fs_raw=250):
    rr = [] #用来存储RR间期的数组
    R_pot = [] #用来存储满足条件R点的数组
    for index,value in enumerate(signal):
        if(index>0):
            rr_pot = (value-signal[index-1])/fs_raw
            #去除异常的RR间期数值，保留[0.3,1.5]区间的数值
            if(rr_pot < 1.5 and rr_pot >0.3):
                rr.append(rr_pot)
                R_pot.append(value)
    return rr,R_pot
# 线性内插函数
def Interplt(rr, R_pot,fs=250):            
    x = np.array(R_pot)/fs
    y = rr
    tck = interpolate.splrep(x, y, s=0)
    t = 8*60 # 8min片段
    xnew = np.arange(0,t,0.5)
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew,ynew

# 将睡眠标签及8分钟观察数据追加进数组data_for_train
def write_tag_observation(file,a_refer=9):
    # 计算有效区间
    obsv_rr = RR_final(file)
    a = valid(obsv_rr) #file的有效区间
    
    # 计算RR间期
    record = wfdb.rdrecord(file, channels=[0])
    ann_ecg = wfdb.rdann(file,'ecg')
    ann_st = wfdb.rdann(file,'st')

    # print(ann_st.aux_note[:10])
    st = [] #用来存储睡眠状态标签的数组
    for aux_note in ann_st.aux_note:
        st.append(aux_note.split(" ")[0])
    fs = record.fs       

    # 将MT（Moment time 运动状态替换为W）
    for index, value in enumerate(st):
        if value=='MT':
            st[index]='W'

    # slp01a 共有len(st)=240个睡眠分期，因为选取的RIS序列为包含该睡眠分期的T=8min片段，因此可训练睡眠分期共有240-15=225个，即st[15:len(st)]
    # 观察值序列分别为[0,8min],[0.5,8.5min],[1,9min]...共225个片段对应的经过插值滤波后的8分钟RR间期，下面给出实现代码

    T = 8
#     fs = 250
    min_sample = 60*fs

    data_for_train = []
    for st_num in range(len(st)-2*T+1):
        sig_start = 0.5*st_num*min_sample
        sig_end = (T+0.5*st_num)*min_sample
        hmm_rr_st_index = []
        for index,value in enumerate(ann_ecg.sample):
            if(value >= sig_start and value <= sig_end):
                hmm_rr_st_index.append(index)
        hmm_rr_st = ann_ecg.sample[hmm_rr_st_index[0]:hmm_rr_st_index[-1]] # 每个睡眠分期对应的T=8分钟RR间期索引序列

        # 获取RR间期序列，并重采样
        st_rr,st_R_pot = RR(hmm_rr_st,fs)
        
        st_R_pot = st_R_pot-st_R_pot[0]
        hmm_rr_st = Interplt(st_rr,st_R_pot)[1] #三次样条插值函数Interplt
        
        hmm_rr_st_num = hmm_rr_st - signal.medfilt(hmm_rr_st,101) #将滤波后的RR间期序列赋给对应睡眠分期的观测值
        hmm_obsv_rr = []
        #限幅和编码
        for value in hmm_rr_st_num:
            if(value < -0.3):
                value = -37 #int(-0.3*1000/8)
            elif(value > 0.3):
                value = 37 #int(0.3*1000/8)
            else:
                value = int(1000*value/8)
            value = int(value*a_refer/a) #标准化
            hmm_obsv_rr.append(value)

        data_for_train.append([st[st_num+2*T-1],hmm_obsv_rr]) #将睡眠标签及对应8分钟观察值存进数组data_for_train
    return data_for_train

# 存储滤波、限幅编码后的RR间期序列，为通过直方图匹配变换来消除不同个体RR间期差异做准备
def RR_final(file):
    # 计算RR间期
    record = wfdb.rdrecord(file, channels=[0])
    ann_ecg = wfdb.rdann(file,'ecg')
    ann_st = wfdb.rdann(file,'st')

    # print(ann_st.aux_note[:10]）
    st = [] #用来存储睡眠状态标签的数组
    for aux_note in ann_st.aux_note:
        st.append(aux_note.split(" ")[0])
    fs = record.fs
    rr,r_pot = RR(ann_ecg.sample,fs)          

    # 将MT（Moment time 运动状态替换为W）
    for index, value in enumerate(st):
        if value=='MT':
            st[index]='W'
            
    
    #三次样条插值函数Interplt
    x = np.array(r_pot)/record.fs
    y = rr
    tck = interpolate.splrep(x, y, s=0)
    t = record.sig_len/record.fs      #信号长度（s）
    xnew = np.arange(0,t,0.5)
    rr_new = interpolate.splev(xnew, tck, der=0)
        
        
    rr_filt = rr_new - signal.medfilt(rr_new,101) #将滤波后的RR间期序列赋给对应睡眠分期的观测值
    obsv_rr = []
    #限幅和编码
    for value in rr_filt:
        if(value < -0.3):
            value = -37 #int(-0.3*1000/8)
        elif(value > 0.3):
            value = 37 #int(0.3*1000/8)
        else:
            value = int(1000*value/8)
        obsv_rr.append(value)

    return obsv_rr

# 为消除不同个体差异应用直方图匹配法：先寻找hmm_obsv_rr的90%有效区间[-a,a]
def valid(obsv_rr):
    for a in range(37):
        count_a = 0 # 将介于区间[-a,a]的r计数，初始化为0
        for r in obsv_rr:
            if(r > -a and r < a):
                count_a += 1
        if(count_a > 0.9*len(obsv_rr)):
            choose_a = count_a
            break
    return a # 即90%的点落在[-a,a]之间    
# 将睡眠标签转换为list(range(6))
def st2num(st):
    stations = ['1','2','3','4','R','W']
    Dic_st = {stations[i]:i for i in range(6)}
    return Dic_st[st]
# 将睡眠标签转换为net格式
def vectorized_result(j):
    e = np.zeros((6, 1))
    e[j] = 1.0
    return e
