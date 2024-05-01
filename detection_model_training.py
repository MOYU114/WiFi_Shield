import csv

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import utils
import utils_model
from sklearn.preprocessing import OneHotEncoder
#from keras.utils import to_categorical
#path
CSI_test = "./data/csi_result_2.4m_apartment_c200/train.csv"
Video_test = "./data/static/points_static.csv"
MODEL_PATH = "./model/"
STUDENT = "student_model.pth"
IDENTIFY= "identify_model.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("load student model.")
ev_latent_dim = 64
es_input_dim = 10
es_hidden_dim = 300
dv_output_dim = 28
id_output_dim=3 #最后的标签数量

#model
student_model = utils_model.StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim).to(device)
student_model.load_state_dict(torch.load(MODEL_PATH+STUDENT))
identify_model=utils_model.identifyModel(dv_output_dim,id_output_dim).to(device)

learning_rate = 0.01
beta1 = 0.5
beta2 = 0.999
optimizer = torch.optim.Adam(identify_model.parameters(), lr=learning_rate, betas=(beta1, beta2))
criterion1 = nn.MSELoss()



with open(CSI_test, "r") as csvfilee:
    csvreadere = csv.reader(csvfilee)
    data2 = list(csvreadere)  # 将读取的数据转换为列表
csi_test = pd.DataFrame(data2)
y_label=csi_test.iloc[:,-1].values
csi_test=csi_test.iloc[:,0:50]
csi_tmp = utils.reshape_and_average(csi_test)
csi_tmp2 = utils.csi_data_preprocess(csi_tmp)

b = torch.from_numpy(csi_tmp2).double()

data = np.hstack((b, y_label.reshape(-1,1)))
np.random.shuffle(data)
train_size = int(len(data) * 0.7)
test_size = len(data) - train_size
csi_train=data[0:train_size,:50]
csi_test=data[train_size:,:50]
train_label=data[0:train_size,-1]
test_label=data[train_size:,-1]

csi_train = csi_train.reshape(len(csi_train), int(len(csi_train[0]) / 10), 10)
csi_train = torch.from_numpy(csi_train.astype(float)).to(device)
encoder = OneHotEncoder()
y = encoder.fit_transform(train_label.reshape(-1, 1)).toarray()
y = torch.from_numpy(y).to(device)

num_epochs=1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    r=student_model(csi_train)
    r=r.reshape(len(csi_train),28)
    res = identify_model(r)
    total_loss = criterion1(y,res)
    # 计算梯度
    total_loss.backward()
    # 更新模型参数
    optimizer.step()

    # 打印训练信息
    print(f"Total Loss: {total_loss.item():.4f}")
print("save student model.")
torch.save(student_model.state_dict(), MODEL_PATH+IDENTIFY)

print("test model.")
csi_test = csi_test.reshape(len(csi_test), int(len(csi_test[0]) / 10), 10)
csi_test = torch.from_numpy(csi_test.astype(float)).to(device)
encoder = OneHotEncoder()
y_test = encoder.fit_transform(test_label.reshape(-1, 1)).toarray()
y_test= torch.from_numpy(y_test).to(device)
with torch.no_grad():
    tr = student_model(csi_test)
    tr = tr.reshape(len(csi_test), 28)
    tres=identify_model(tr)
    #r = r.view(np.size(r, 0), np.size(r, 1))  # CNN
    loss = criterion1(y_test, tres)
    # 打印训练信息
    print(f"Test Loss: {loss.item():.4f}")