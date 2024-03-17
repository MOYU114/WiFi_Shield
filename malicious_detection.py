import logging
import csv
import os
from tqdm import tqdm,trange
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import utils
import utils_model
import matplotlib.pyplot as plt
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# param
ev_input_dim = 28
ev_latent_dim = 64
es_input_dim = 10
es_hidden_dim = 300
dv_output_dim = 28
threshold = 400
lamb=11
# path
CSI_PATH = "./data/static/CSI_static_6C.csv"
Video_PATH = "./data/static/points_static.csv"
MODEL_OUTPUT_PATH = "./model/"
CSI_AVG_PATH ="./data/static/CSI_static_6C_avg.csv"
LOG_PATH="./output/log.txt"
UPDATE = True
Alarm = False
# 假定这是经过了固定的时间T收集的到的CSI数据
with open(CSI_PATH, "r", encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    data1 = list(csvreader)
aa = pd.DataFrame(data1)  # 读取CSI数据到aa

#需要一个时间函数，去判断是否到了更新数据的阶段
if UPDATE:
    # calculate CSI_avg
    print("update threshold.")
    '''
    if (os.path.exists(CSI_AVG_PATH) != True):
        CSI_avg = utils.cal_avg(aa)
        np.savetxt(CSI_AVG_PATH, CSI_avg, delimiter=',')
    else:
        with open(CSI_AVG_PATH, "r", encoding='utf-8-sig') as csvfile:
            csvreader = csv.reader(csvfile)
            data1 = list(csvreader)  # 将读取的数据转换为列表
        CSI_avg = pd.DataFrame(data1)
    #CSI_avg = utils.reshape_and_average(aa)  # 把多个CSI数据包平均为一个数据包，使一帧对应一个CSI数据包
    '''
    CSI_avg = utils.cal_avg(aa)
    CSI_avg = CSI_avg.T.squeeze()
    CSI_avg = CSI_avg.values.astype('float32')
    print("get CSI_avg")
    CSI_avg_median = np.median(CSI_avg)
    CSI_avg_mad = np.median(np.abs(CSI_avg - np.median(CSI_avg)))
    threshold= lamb*CSI_avg_mad+CSI_avg_median

# 假设这是一秒内实时接收到的数据，假设15帧/s

crr_CSI= utils.cal_single_avg(aa[:15])
Alarm=True
#Alarm=crr_CSI>threshold




if(Alarm):
    begin_t = int(round(time.time() * 1000))
    begin = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(begin_t / 1000))
    #while(crr_CSI>threshold):
    #    now = int(round(time.time() * 1000))
    time.sleep(1)
    now = int(round(time.time() * 1000))
    end = time.strftime('%H:%M:%S', time.localtime(now / 1000))
    Alarm = False
    logger = utils.logger_config(log_path=LOG_PATH, logging_name='test')
    logger.warning(begin+"~"+end+" 检测到有人经过。")

tt=CSI_avg.squeeze()
plt.plot(np.arange(0,4800),np.full(4800,threshold))
plt.plot(np.arange(0,4800),tt[:4800])
plt.show()