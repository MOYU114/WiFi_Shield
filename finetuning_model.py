import csv

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import utils
import utils_model

#path
Video_test = "./data/static/data/device/points_arm_left.csv"
CSI_test = "./data/static/data/device/CSI_arm_left_6C_1.csv"
CSI_OUTPUT_PATH = "./data/output/CSI_loc1.csv"
Video_OUTPUT_PATH = "./data/output/points_merged_output1.csv"
MODEL_PATH = "./model/xxx.pth"

with open(CSI_test, "r") as csvfilee:
    csvreadere = csv.reader(csvfilee)
    data2 = list(csvreadere)  # 将读取的数据转换为列表
csi_test = pd.DataFrame(data2)
video_test = pd.read_csv(Video_test, header=None)
print("test data has loaded.")

csi_tmp = utils.reshape_and_average(csi_test)
csi_tmp2 = utils.csi_data_preprocess(csi_tmp)
video_tmp2 = utils.video_data_preprocess(video_test)

#model

criterion1 = nn.MSELoss()


ev_latent_dim = 64
es_input_dim = 10
es_hidden_dim = 300
dv_output_dim = 28
student_model = utils_model.StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim)
student_model.load_state_dict(torch.load(MODEL_PATH))

# CNN
g = torch.from_numpy(video_tmp2).double()
b = torch.from_numpy(csi_tmp2).double()
b = b.view(len(b), int(len(csi_tmp2[0]) / 10), 10)
# 未修改
with torch.no_grad():
    r = student_model(b)
    r = r.view(np.size(r, 0), np.size(r, 1))  # CNN
    loss = criterion1(r, g)
    print("loss:", loss)
    g = g.cpu()
    r = r.cpu()
    gnp = g.numpy()
    rnp = r.numpy()
    np.savetxt(Video_OUTPUT_PATH, gnp, delimiter=',')
    np.savetxt(CSI_OUTPUT_PATH, rnp, delimiter=',')
