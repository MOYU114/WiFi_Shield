import csv
import math
import os
import torch
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
    id_output_dim = 5  # 最后的标签数量
    threshold = 400
    lamb = 2
    minute=5
    begin_t=0
    t_threshold=15
    # path
    HISTORY_CSI_LOG_PATH = "./data/static/CSI_static_6C.csv"
    CRR_CSI_PATH = "./data/static/CSI_new.csv"
    MODEL_PATH = "./model/"
    LOG_PATH = "./output/log.txt"
    IDENTIFY="identify_model_5.pth"
    STUDENT="student_model.pth"
    # sign
    UPDATE=True
    DATA_AVAILABLE=True
    threadLock = threading.Lock()

    dic = ["left_arm", "left_leg", "open", "right_arm", "right_leg"]
    #dic=["left_arm","right_arm","stand"]
    def __init__(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.begin_t = int(round(time.time() * 1000))#需要跟CSI同步时间

    def __readCSI(self,csi_path):
        with open(csi_path, "r", encoding='utf-8-sig') as csvfile:
            csvreader = csv.reader(csvfile)
            data1 = list(csvreader)
        aa = pd.DataFrame(data1)  # 读取CSI数据到aa

        return aa

    def __updateThreshold(self, aa):
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
        #CSI_avg = CSI_avg.T.squeeze()
        #CSI_avg = CSI_avg.values.astype('float32')
        print("get CSI_avg")
        CSI_avg_median = np.median(CSI_avg)
        #MAD对于数据集中异常值的处理比标准差更具有弹性，可以大大减少异常值对于数据集的影响
        CSI_avg_mad = np.median(np.abs(CSI_avg - np.median(CSI_avg)))
        #CSI_std=np.std(CSI_avg)
        self.threshold = self.lamb * CSI_avg_mad + CSI_avg_median
    def __updateHistoryLog(self, CRR_CSI_PATH, HISTORY_CSI_LOG_PATH):
        aa = self.__readCSI(HISTORY_CSI_LOG_PATH)
        bb = self.__readCSI(CRR_CSI_PATH)
        aa=pd.concat([aa,bb], ignore_index = True)
        # 之后考虑5min内收集多少行数据，超过该数据则要对history数据清理
        aa.to_csv(HISTORY_CSI_LOG_PATH, header=None, index=None)

    def __isAlarm(self, crr_aa):
        begin=[]
        end=[]
        S = utils.cal_avg(crr_aa)
        i =0
        while(i<len(S)):
            if(S[i]>=self.threshold):
                j=i
                while(j<len(S) and S[j]>=self.threshold ):
                    j+=1
                begin.append(i)
                end.append(j)
                i=j
            i+=1
        return begin,end
    def __getPoseInfo(self, begin, end, csi):

        # model
        student_model = utils_model.StudentModel(self.dv_output_dim, self.es_input_dim, self.es_hidden_dim, self.ev_latent_dim).to(self.device)
        student_model.load_state_dict(torch.load(self.MODEL_PATH + self.STUDENT))
        identify_model = utils_model.identifyModel(self.dv_output_dim, self.id_output_dim).to(self.device)
        identify_model.load_state_dict(torch.load(self.MODEL_PATH+self.IDENTIFY))
        id_csi = csi.values
        id_csi = id_csi[begin:end,0:50]
        id_csi = id_csi.reshape(len(id_csi), int(len(id_csi[0]) / 10), 10)
        id_csi = torch.from_numpy(id_csi.astype(float)).to(self.device)
        tr = student_model(id_csi)
        tr = tr.reshape(len(id_csi), 28)
        tres = identify_model(tr)
        pos = tres.argmax(dim=1)
        pos = pos.cpu()
        counts = np.bincount(pos)
        # 返回前三个出现频率最多的数
        #possible_pos=np.argmax(counts)
        possible_pos = np.argsort(counts)
        #tres_0 = torch.mean(tres, dim=0)
        #possible_pos=tres_0.argmax()
        #counts_sum=np.sum(counts)
        return possible_pos,counts

    def __alarm(self, csi, begin, end):
        #需要跟CSI同步时间
        time_sq=[]
        time_seq=1000
        logger = utils.logger_config(log_path=self.LOG_PATH, logging_name='test')
        for i in range(len(begin)):
            if(end[i]-begin[i]<=self.t_threshold):#驻留时间
                continue
            #seq=end[i]-begin[i]
            # while(crr_CSI>threshold):
            #    now = int(round(time.time() * 1000))
            begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime((self.begin_t+begin[i]*time_seq) / 1000))
            end_time = time.strftime('%H:%M:%S', time.localtime((self.begin_t+(end[i]-1)*time_seq) / 1000))
            possible_pos,counts=self.__getPoseInfo(begin[i], end[i], csi)
            res_str = self.dic[possible_pos[-1]] + ":" + str(round(counts[possible_pos[-1]] / np.sum(counts), 2)) + " " \
                      + self.dic[possible_pos[-2]] + ":" + str(round(counts[possible_pos[-2]] / np.sum(counts), 2)) + " " \
                      + self.dic[possible_pos[-3]] + ":" + str(round(counts[possible_pos[-3]] / np.sum(counts), 2)) + " "


            logger.warning(begin_time + "~" + end_time + " 检测到有人"+self.dic[possible_pos[-1]]+ " 前三动作占比为"+res_str)
            self.t_threshold=0.8*self.t_threshold+0.2*(end[i]-begin[i])

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
                aa = self.__readCSI(self.HISTORY_CSI_LOG_PATH)
                self.__updateThreshold(aa)
                #lock
                self.threadLock.acquire()
                self.UPDATE = False
                self.threadLock.release()

            if(self.DATA_AVAILABLE):#有新数据传入
                crr_aa = self.__readCSI(self.NEW_CSI_PATH)
                begin,end=self.__isAlarm(crr_aa)
                if (len(begin) > 0):
                    self.__alarm(crr_aa, begin, end)
                else:
                    #将当前数据传入历史记录中
                    self.__updateHistoryLog(self.CRR_CSI_PATH, self.HISTORY_CSI_LOG_PATH)
            now = int(round(time.time() * 1000))
            end = time.strftime('%H:%M:%S', time.localtime(now / 1000))
    def draw_plt(self,a):
        X=[]
        for i in range(0,600-15,5):
            X.append(i)
        plt.plot(X,a[3:])
        plt.plot()
        plt.vlines([185, 385], 0, 0.35, linestyles='dashed', colors='red')
        plt.xlabel("time/s")
        plt.ylabel("$\overline{S}$")
        plt.show()
    def test(self,normal,warning):
        a_nor=self.__readCSI(normal)
        a_war = self.__readCSI(warning)
        self.__updateThreshold(a_nor)
        begin,end = self.__isAlarm(a_war)
        if (len(begin)>0):
            self.__alarm(a_war, begin, end)

if __name__ == '__main__':
    TEST=True
    if TEST:
        #warning="./data/left_arm.csv"
        warning="./data/example.csv"
        normal="./data/empty.csv"
        #csi_result_meeting_room
        test=detection_proc()
        test.test(normal,warning)
    else:
        dec=detection_proc()
        dec.run()

