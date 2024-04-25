import math
import csv
import os
import logging
from tqdm import tqdm,trange
import torch, gc
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.init as init
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

#two ways to handle data
def reshape_and_average(x):
    num_rows = x.shape[0]
    averaged_data = np.zeros((num_rows, 50))
    for i in trange(num_rows):
        row_data = x.iloc[i].to_numpy()
        reshaped_data = row_data.reshape(-1, 50)
        reshaped_data = pd.DataFrame(reshaped_data).replace({None: np.nan}).values
        reshaped_data = pd.DataFrame(reshaped_data).dropna().values
        non_empty_rows = np.any(reshaped_data != '', axis=1)
        filtered_arr = reshaped_data[non_empty_rows]
        reshaped_data = np.asarray(filtered_arr, dtype=np.float64)
        averaged_data[i] = np.nanmean(reshaped_data, axis=0)  # Compute column-wise average
    averaged_df = pd.DataFrame(averaged_data, columns=None)
    return averaged_df

def fillna_with_previous_values(s):
    non_nan_values = s[s.notna()].values
    nan_indices = s.index[s.isna()]
    n_fill = len(nan_indices)
    n_repeat = int(np.ceil(n_fill / len(non_nan_values)))
    fill_values = np.tile(non_nan_values, n_repeat)[:n_fill]
    s.iloc[nan_indices] = fill_values
    return s


# 把静止的数据的某一类提取复制出对应数目的静态骨架帧
def gene_static(index, num):
    raw_data_file = "./data/static/points_left_right_leg_stand_sit.csv"
    raw_data = pd.read_csv(raw_data_file, header=None)

    if index == 1:
        arm_left = raw_data.iloc[0]
        arm_left_data = pd.DataFrame(np.tile(arm_left, (num, 1)))
        return arm_left_data
    elif index == 2:
        arm_right = raw_data.iloc[1]
        arm_right_data = pd.DataFrame(np.tile(arm_right, (num, 1)))
        return arm_right_data
    elif index == 3:
        leg = raw_data.iloc[2]
        leg_data = pd.DataFrame(np.tile(leg, (num, 1)))
        return leg_data
    elif index == 4:
        leg = raw_data.iloc[3]
        stand_data = pd.DataFrame(np.tile(leg, (num, 1)))
        return stand_data
    else:
        stand = raw_data.iloc[4]
        sit_data = pd.DataFrame(np.tile(stand, (num, 1)))
        # stand_data.to_csv('temp.csv', index=False) %把数据存储到csv文件
        return sit_data


# 通过关节点的制约关系得到wave，leg和stand的索引，然后返回相同数量的三种类别的索引
def group_list(frame_value):
    leg_index = []
    wave_index = []
    stand_index = []

    for i in range(len(frame_value)):
        if frame_value[i, 9] - frame_value[i, 5] < 50:
            wave_index.append(i)
        elif frame_value[i, 26] - frame_value[i, 20] > 160:
            leg_index.append(i)
        elif frame_value[i, 26] - frame_value[i, 20] < 100 and frame_value[i, 9] - frame_value[i, 5] > 150:
            stand_index.append(i)
        else:
            continue

    length_min = min(len(wave_index), len(leg_index), len(stand_index))
    leg_index = leg_index[0:length_min * 8]
    wave_index = wave_index[0:length_min * 8]
    stand_index = stand_index[0:length_min]
    merged_index = leg_index + wave_index + stand_index
    return merged_index


def poseloss(real, fake):
    # 生成权重值数组
    weights = torch.zeros((28,))
    indices = [6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21, 24, 25, 26, 27]
    weights[indices] = 1.0

    # 将 Tensor 转换为一维张量
    real_flat = real.view(real.size(0), -1)
    fake_flat = fake.view(fake.size(0), -1)

    # 计算两个 Tensor 对应位置的差值的平方
    squared_diff = (real_flat - fake_flat) ** 2

    # 分别乘以权重值
    weighted_diff = squared_diff * weights

    # 求和
    result = torch.sum(weighted_diff, dim=1)  # 沿着第二个维度求和
    result_mean = torch.mean(result)
    return result_mean

# 随机初始化生成器和鉴别器的参数
def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)  # 使用Xavier均匀分布初始化权重
        if m.bias is not None:
            init.constant_(m.bias.data, 0.1)  # 初始化偏置为0.1

def video_data_preprocess(ff):
    Video_train = ff.values.astype('float32')
    Video_train = Video_train.reshape(len(Video_train), 14, 2)  # 分成990组14*2(x,y)的向量
    Video_train = Video_train / [1280, 720]  # 输入的图像帧是1280×720的，所以分别除以1280和720归一化。
    Video_train = Video_train.reshape(len(Video_train), -1)
    return Video_train
def csi_data_preprocess(bb):
    CSI_train = bb.values.astype('float32')
    CSI_train = CSI_train / np.max(CSI_train)
    return CSI_train
'''
def cal_avg(alp_CSI):
    num_rows = alp_CSI.shape[0]
    averaged_data = np.zeros((num_rows, 1))
    for i in trange(num_rows):
        row_data = alp_CSI.iloc[i].to_numpy().astype('float64')
        averaged_data[i] = np.nanmean(row_data)
    return averaged_data
'''
def old_cal_avg(x):
    num_rows = x.shape[0]
    averaged_data = np.zeros((num_rows, 1))
    for i in trange(num_rows):
        row_data = x.iloc[i].to_numpy()
        reshaped_data = row_data.reshape(-1, 1)
        reshaped_data = pd.DataFrame(reshaped_data).replace({None: np.nan}).values
        reshaped_data = pd.DataFrame(reshaped_data).dropna().values
        non_empty_rows = np.any(reshaped_data != '', axis=1)
        filtered_arr = reshaped_data[non_empty_rows]
        reshaped_data = np.asarray(filtered_arr, dtype=np.float64)
        averaged_data[i] = np.nanmean(reshaped_data, axis=0)  # Compute column-wise average
    averaged_df = pd.DataFrame(averaged_data, columns=None)
    return averaged_df

def cal_avg(x):

    num_rows = x.shape[0]
    averaged_data = np.zeros((num_rows, 1))
    for i in trange(num_rows):
        row_data = x.iloc[i].to_numpy()
        reshaped_data = row_data.reshape(-1, 1)
        reshaped_data = pd.DataFrame(reshaped_data).replace({None: np.nan}).values
        reshaped_data = pd.DataFrame(reshaped_data).dropna().values
        non_empty_rows = np.any(reshaped_data != '', axis=1)
        filtered_arr = reshaped_data[non_empty_rows]
        reshaped_data = np.asarray(filtered_arr, dtype=np.float64)
        averaged_data[i] = np.nanmean(reshaped_data, axis=0)  # Compute column-wise average
    averaged_df = pd.DataFrame(averaged_data, columns=None)
    S = (averaged_df / np.max(averaged_df)).applymap(lambda x: 1 - x)
    S = S.values

    seq_size = 10
    S_num_rows= len(S) // seq_size + (1 if len(S) % seq_size else 0)
    S = S.reshape(S_num_rows, -1)
    S_averaged_data = np.zeros((S_num_rows, 1))
    for i in trange(S_num_rows):
        S_averaged_data[i] = np.average(S[i])
    return S_averaged_data
def cal_single_avg(x):
    num_rows = x.shape[0]
    averaged_data = np.zeros((num_rows, 1))
    for i in range(num_rows):
        row_data = x.iloc[i].to_numpy()
        reshaped_data = row_data.reshape(-1, 1)
        reshaped_data = pd.DataFrame(reshaped_data).replace({None: np.nan}).values
        reshaped_data = pd.DataFrame(reshaped_data).dropna().values
        non_empty_rows = np.any(reshaped_data != '', axis=1)
        filtered_arr = reshaped_data[non_empty_rows]
        reshaped_data = np.asarray(filtered_arr, dtype=np.float64)
        averaged_data[i] = np.nanmean(reshaped_data, axis=0)  # Compute column-wise average
    averaged_df = pd.DataFrame(averaged_data, columns=None)
    return np.nanmean(averaged_df.iloc[0].to_numpy().astype('float64'))

def logger_config(log_path,logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    使用：
    logger = logger_config(log_path='xxx', logging_name='xxx')
    logger.info("info")
    logger.error("error")
    logger.debug("debug")
    logger.warning("warning")
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger