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
import threading

class detection_proc:

    device = torch.device('cpu')
    # param
    ev_input_dim = 28
    ev_latent_dim = 64
    es_input_dim = 10
    es_hidden_dim = 300
    dv_output_dim = 28
    threshold = 400
    lamb = 11
    minute=5
    # path
    HISTORY_CSI_LOG_PATH = "./data/static/CSI_static_6C.csv"
    CRR_CSI_PATH = "./data/static/CSI_new.csv"
    MODEL_OUTPUT_PATH = "./model/"
    CSI_AVG_PATH = "./data/static/CSI_static_6C_avg.csv"
    LOG_PATH = "./output/log.txt"
    # sign
    UPDATE=True
    threadLock = threading.Lock()
    def __init__(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def __readCSI(self,csi_path):
        with open(csi_path, "r", encoding='utf-8-sig') as csvfile:
            csvreader = csv.reader(csvfile)
            data1 = list(csvreader)
        aa = pd.DataFrame(data1)  # 读取CSI数据到aa
        return aa

    def updateThreshold(self,aa):
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
        self.threshold = self.lamb * CSI_avg_mad + CSI_avg_median
    def updateHistoryLog(self,CRR_CSI_PATH,HISTORY_CSI_LOG_PATH):
        aa = self.__readCSI(HISTORY_CSI_LOG_PATH)
        bb = self.__readCSI(CRR_CSI_PATH)
        aa=pd.concat([aa,bb], ignore_index = True)
        # 之后考虑5min内收集多少行数据，超过该数据则要对history数据清理
        aa.to_csv(HISTORY_CSI_LOG_PATH, header=None, index=None)

    def isAlarm(self,crr_aa):

        crr_CSI = utils.cal_single_avg(crr_aa)
        flag = crr_CSI > self.threshold
        return crr_CSI, flag

    def alarm(self,crr_CSI):
        begin_t = int(round(time.time() * 1000))
        begin = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(begin_t / 1000))
        # while(crr_CSI>threshold):
        #    now = int(round(time.time() * 1000))
        time.sleep(1)
        now = int(round(time.time() * 1000))
        end = time.strftime('%H:%M:%S', time.localtime(now / 1000))

        logger = utils.logger_config(log_path=self.LOG_PATH, logging_name='test')
        logger.warning(begin + "~" + end + " 检测到有人经过。")
    def __m2s(self,minute):
        return minute*60
    def __canUpdate(self,min):
        while(1):
            print("running update")
            #begin_time = time.time()
            time.sleep(self.__m2s(min))
            #crr_time = time.time()
            #print(int(crr_time-begin_time))
            #lock
            self.threadLock.acquire()
            self.UPDATE = True
            self.threadLock.release()

    def run(self):
        #单开一个线程计时
        t1 = threading.Thread(target=self.__canUpdate, args=(self.minute,))
        t1.start()
        while(1):
            if(self.UPDATE):#每五分钟更新一次
                aa = self.__readCSI(HISTORY_CSI_LOG_PATH)
                self.updateThreshold(aa)
                #lock
                self.threadLock.acquire()
                self.UPDATE = False
                self.threadLock.release()

            if(False):#有新数据传入
                crr_aa = self.__readCSI(NEW_CSI_PATH)
                crr_CSI,flag=self.isAlarm(crr_aa)
                if(flag):
                    self.alarm(crr_CSI)
                else:
                    #将当前数据传入历史记录中
                    self.updateHistoryLog(self.CRR_CSI_PATH,self.HISTORY_CSI_LOG_PATH)
            now = int(round(time.time() * 1000))
            end = time.strftime('%H:%M:%S', time.localtime(now / 1000))
            print(end)
    def test(self,normal,warning):
        a_nor=self.__readCSI(normal)
        a_war = self.__readCSI(warning)
        self.updateThreshold(a_nor[:1000])
        crr_CSI, flag = self.isAlarm(a_war[:1000])
        if (flag):
            self.alarm(crr_CSI)

if __name__ == '__main__':
    warning="./data/CSI_warning.csv"
    normal="./data/CSI_normal.csv"
    test=detection_proc()

    test.test(normal,warning)
