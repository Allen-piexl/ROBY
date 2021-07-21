# -*- coding: utf-8 -*-
"""
模型提取特征的统计，包括几个指标：
FSA: 同类数据特征子空间聚合度
FSD: 不同类特征子空间距离
FSC: 特征子空间重合度
输入表示提取特征的数组，计算统计值。

"""
import numpy as np
import tensorflow as tf
import model_lib as lib
import math
#import cifar10_input
import os
from tensorflow.examples.tutorials.mnist import input_data
# 距离函数，用欧式距离
def dist(point, center):
    d = np.sqrt(np.sum(np.square(point - center)))
    return d

def feature_sta(features_list):
    # FSA 同类数据特征子空间聚合度
    d_list = []
    center_list = []
    for i in range(len(features_list)):
        center = np.mean(features_list[i], axis=0)
        center_list.append(center)
        d_list.append(np.mean([dist(point, center) for point in features_list[i]]))
    # 归一化
    norm_d_list = []
    for i in range(len(d_list)):
        d = (d_list[i] - np.min(d_list))/(np.max(d_list) - np.min(d_list))
        norm_d_list.append(d)
    FSA = 1 - np.mean(norm_d_list)

    # FSD 不同类特征子空间距离
    d2_list = []
    for i in range(len(center_list)-1):
        for j in range(i+1, len(center_list)):
            d2 = dist(center_list[i], center_list[j])
            d2_list.append(d2)
            d2d = d2/(d_list[i]+d_list[j])
    # 归一化
    norm_d2_list = []
    for i in range(len(d2_list)):
        d = (d2_list[i] - np.min(d2_list))/(np.max(d2_list) - np.min(d2_list))
        norm_d2_list.append(d)
    FSD = np.mean(norm_d2_list)

    # FSC 特征子空间重合度
    FSC_list = []
    for i in range(len(d_list)):
        for j in range(i+1, len(d_list)):
            FSC_d = d_list[i] + d_list[j] - dist(center_list[i], center_list[j])
            FSC_list.append(FSC_d)
    # 归一化
    norm_FSC_list = []
    for i in range(len(FSC_list)):
        d = (FSC_list[i] - np.min(FSC_list)) / (np.max(FSC_list) - np.min(FSC_list))
        norm_FSC_list.append(d)
    FSC = np.mean(norm_FSC_list)

    return FSA, FSD, FSC

# 提取神经网络倒数第二层特征，一个样本展成一维并存在数组中，一行代表一类
model = lib.MobileNet(is_training=True)
data_path = '/public/home/cjy/Documents/Dataset/DNN/fashion_data'
model_dir = '/public/home/cjy/Documents/Python/DNN/code-mnist/mobilenet_eval/models-f/robust_mobilenet'
cur_checkpoint = tf.train.latest_checkpoint(model_dir)
# mnist数据集
mnist = input_data.read_data_sets(data_path, one_hot=False)
data_num = len(mnist.test.images)
batch_size = 50
class_num = 10

batch_num = int(math.ceil(data_num / batch_size))

feature_file = [[] for i in range(class_num)]
saver = tf.train.Saver()


os.environ['CUDA_VISIBLE_DEVICES']='3'
gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.allow_growth = True

with tf.Session(config=gpuconfig) as sess:
    # Restore the checkpoint
    saver.restore(sess, cur_checkpoint)
    print("restore checkpoint:{}".format(cur_checkpoint))
    for i in range(batch_num):
        x_batch =  mnist.test.images[i * batch_size:(i + 1) * batch_size, :]
        y_batch =  mnist.test.labels[i * batch_size:(i + 1) * batch_size]
        val_dict = {model.x_input: x_batch,
                    model.y_input: y_batch}

        batch_features = sess.run(model.logit, feed_dict=val_dict)
        # 按一行一类存储，类别从y_batch中获知
        for j in range(len(y_batch)):
            feature_file[y_batch[j]].append(batch_features[j])

FSA, FSD, FSC = feature_sta(feature_file)
print("FSA={}, FSD={}, FSC={}".format(FSA,FSD,FSC))

