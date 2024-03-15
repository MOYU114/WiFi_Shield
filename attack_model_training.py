import csv
from tqdm import tqdm,trange
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import utils
import utils_model

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
# 读取数据
with open(CSI_PATH, "r", encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    data1 = list(csvreader)
aa = pd.DataFrame(data1)  # 读取CSI数据到aa
ff = pd.read_csv(Video_PATH, header=None)  # 读取骨架关节点数据到ff
print("data has loaded.")
print("begin data processing.")
bb = utils.reshape_and_average(aa)  # 把多个CSI数据包平均为一个数据包，使一帧对应一个CSI数据包


Video_train=utils.video_data_preprocess(ff)
CSI_train=utils.csi_data_preprocess(bb)

data = np.hstack((Video_train, CSI_train))
np.random.shuffle(data)
data_length = len(data)
train_data_length = int(data_length * 0.9)
test_data_length = int(data_length - train_data_length)

frame_train = data[0:train_data_length, 0:28]
csi_train = data[0:train_data_length, 28:78]

original_length = frame_train.shape[0]
print("begin to pretrain Teacher model.")
# 预训练Teacher模型
LR_G = 0.001
LR_D = 0.001
teacher_model_G = utils_model.TeacherModel_G(ev_input_dim, ev_latent_dim, dv_output_dim).to(device)
teacher_model_D = utils_model.TeacherModel_D(ev_input_dim).to(device)
criterion1 = nn.MSELoss()
criterion2 = nn.BCELoss()
optimizer_G = torch.optim.Adam(teacher_model_G.parameters(), lr=LR_G)
optimizer_D = torch.optim.Adam(teacher_model_D.parameters(), lr=LR_D)

teacher_model_G.apply(utils.weights_init)
teacher_model_D.apply(utils.weights_init)

torch.autograd.set_detect_anomaly(True)
Teacher_num_epochs = 1300
teacher_batch_size = 128
#epoch_losses0 = []
#epoch_losses1 = []

for epoch in range(Teacher_num_epochs):
    random_indices = np.random.choice(original_length, size=teacher_batch_size, replace=False)
    f = torch.from_numpy(frame_train[random_indices, :]).double()
    f = f.view(teacher_batch_size, 28, 1, 1)  # .shape(batch_size,28,1,1)
    f=f.to(device)
    y = teacher_model_G(f)

    gen_loss = criterion1(y, f)
    gen_loss.backward()
    optimizer_G.step()
    #epoch_losses1.append(gen_loss.item())
    # 打印训练信息
    print(f"TeacherModel training:Epoch [{epoch + 1}/{Teacher_num_epochs}], Teacher_G Loss: {gen_loss.item():.4f}")  # ,Teacher_D Loss: {teacher_loss.item():.4f}")
# 对抗模型训练
print("begin to train TeacherStudent model.")
# 学习率scheduling;
learning_rate = 0.01
beta1 = 0.5
beta2 = 0.999
model = utils_model.TeacherStudentModel(ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
model.teacher_encoder_ev.load_state_dict(teacher_model_G.teacher_encoder_ev.state_dict())
model.teacher_decoder_dv.load_state_dict(teacher_model_G.teacher_decoder_dv.state_dict())
model.teacher_discriminator_c.load_state_dict(teacher_model_D.teacher_discriminator_c.state_dict())

num_epochs =1000
batch_size = 128
for epoch in range(num_epochs):
    random_indices = np.random.choice(original_length, size=batch_size, replace=False)
    f = torch.from_numpy(frame_train[random_indices, :]).double()
    a = torch.from_numpy(csi_train[random_indices, :]).double()
    f = f.view(batch_size, 28, 1, 1)
    a = a.view(batch_size, int(len(csi_train[0]) / 10), 10)
    f = f.to(device)
    a = a.to(device)
    optimizer.zero_grad()
    # f是来自视频帧的Ground Truth，a是幅值帧，z是视频帧embedding，y是视频帧的fake骨架图，v是幅值帧的embedding，s是幅值帧的synthetic骨架帧
    z, y, v, s = model(f, a)
    teacher_loss = criterion1(f,y) + 0.5*criterion1(y,s)
    student_loss =0.6*criterion1(s, f) + criterion1(v, z)
    total_loss = teacher_loss + student_loss
    # 计算梯度
    total_loss.backward()
    # 更新模型参数
    optimizer.step()

    # 打印训练信息
    print(f"training:Epoch [{epoch + 1}/{num_epochs}], Teacher Loss: {teacher_loss.item():.4f}, Student Loss: {student_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
print("save student model.")
student_model = utils_model.StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim)
student_model.selayer.load_state_dict(model.selayer.state_dict())
student_model.student_encoder_es.load_state_dict(model.student_encoder_es.state_dict())
student_model.student_decoder_ds.load_state_dict(model.teacher_decoder_dv.state_dict())
torch.save(student_model.state_dict(), MODEL_OUTPUT_PATH+"student_model.pth")
print("finished!")

