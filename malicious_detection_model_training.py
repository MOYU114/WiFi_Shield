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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# param
ev_input_dim = 28
ev_latent_dim = 64
es_input_dim = 10
es_hidden_dim = 300
dv_output_dim = 28

# path
CSI_PATH = "./data/static/CSI_static_6C.csv"
Video_PATH = "./data/static/points_static.csv"
MODEL_OUTPUT_PATH = "./model/"
CSI_AVG_PATH ="./data/static/CSI_static_6C_avg.csv"
# read CSI
with open(CSI_PATH, "r", encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    data1 = list(csvreader)
aa = pd.DataFrame(data1)  # 读取CSI数据到aa
ff = pd.read_csv(Video_PATH, header=None)  # 读取骨架关节点数据到ff
print("data has loaded.")
# calculate CSI_avg
print("begin data processing.")


if (os.path.exists(CSI_AVG_PATH) != True):
    bb = utils.reshape_and_average(aa)
    np.savetxt(CSI_AVG_PATH, bb, delimiter=',')
else:
    with open(CSI_AVG_PATH, "r", encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile)
        data1 = list(csvreader)  # 将读取的数据转换为列表
    bb = pd.DataFrame(data1)
#CSI_avg = utils.reshape_and_average(aa)  # 把多个CSI数据包平均为一个数据包，使一帧对应一个CSI数据包
CSI_avg = utils.cal_avg(bb)

print("get CSI_avg")
tt=CSI_avg.squeeze()
print(tt)
plt.plot(np.arange(0,100),tt[:100])
plt.show()