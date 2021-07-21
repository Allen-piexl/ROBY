# -*- coding: utf-8 -*-
"""
@author: Zhen-Wang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
#import cifar10_input
import model_lib as lib
from pgd_attack import LinfPGDAttack
from tensorflow.examples.tutorials.mnist import input_data
import math

tf.set_random_seed(12345)
np.random.seed(54321)

eval_batch_size = 200
num_eval_examples=10000

#model_dir = 'models/natural_alexnet'
data_path = '/home/zhb/wangzhen/Documents/Dataset/DNN/fashion_data'
model_dir = '/home/zhb/wangzhen/Documents/Python/DNN/code-mnist/mobilenet_eval/models-f/natural_mobilenet'
cur_checkpoint = tf.train.latest_checkpoint(model_dir)
mnist = input_data.read_data_sets(data_path, one_hot=False)

model = lib.MobileNet(is_training=True)
saver = tf.train.Saver()

os.environ['CUDA_VISIBLE_DEVICES']='1'
gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.allow_growth = True

# 对原始样本的测试
with tf.Session(config=gpuconfig) as sess:

    batch_num = int(len(mnist.test.images) / eval_batch_size)
    total_correct = 0

    saver.restore(sess, cur_checkpoint)
    print("restore checkpoint:{}".format(cur_checkpoint))
    for ii in range(batch_num):
        x_batch = mnist.test.images[ii*eval_batch_size:(ii+1)*eval_batch_size, :]
        y_batch = mnist.test.labels[ii*eval_batch_size:(ii+1)*eval_batch_size]
        val_dict = {model.x_input: x_batch,
                    model.y_input: y_batch}
        num_correct = sess.run(model.num_correct, feed_dict=val_dict)
        total_correct += num_correct
    print("natural test accuracy:{}".format(total_correct/len(mnist.test.images)))


# 对对抗样本的测试1.0
import numpy as np
adv_data_path="/home/zhb/wangzhen/Documents/Python/DNN/code-mnist/mobilenet_eval/fashion-natural-attack.npy"
x_nat = mnist.test.images
x_adv = np.load(adv_data_path)
l_inf = np.amax(np.abs(x_nat - x_adv))
epsilon = 1.0
if l_inf > epsilon + 0.0001:
    print('maximum perturbation found: {}'.format(l_inf))
    print('maximum perturbation allowed: {}'.format(epsilon))

if cur_checkpoint is None:
    print('No checkpoint found')
elif x_adv.shape != (10000, 32, 32, 3):
    print('Invalid shape: expected (10000, 32, 32, 3), found {}'.format(x_adv.shape))
elif np.amax(x_adv) > 255.0001 or np.amin(x_adv) < -0.0001:
    print('Invalid pixel range. Expected [0, 255], found [{}, {}]'.format(
                                                          np.amin(x_adv),
                                                          np.amax(x_adv)))
output, y = model.logit, model.y_input
loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=y))
with tf.Session(config=gpuconfig) as sess:
    batch_num = int(len(x_adv) / eval_batch_size)
    total_correct = 0
    total_loss = 0
    saver.restore(sess, cur_checkpoint)
    print("restore checkpoint:{}".format(cur_checkpoint))
    for ii in range(batch_num):
        x_batch = x_adv[ii*eval_batch_size:(ii+1)*eval_batch_size, :]
        y_batch = mnist.test.labels[ii*eval_batch_size:(ii+1)*eval_batch_size]
        val_dict = {model.x_input: x_batch,
                    model.y_input: y_batch}
        #num_correct = sess.run(model.num_correct, feed_dict=val_dict)
        num_correct, batch_loss = sess.run([model.num_correct, loss], feed_dict=val_dict)
        total_correct += num_correct
        total_loss += batch_loss
    print("adv test accuracy:{}".format(total_correct/len(x_adv)))
    print("adv loss:{}".format(total_loss / batch_num))

"""
# 对对抗样本的测试2.0
x_nat = cifar.eval_data.xs
attack = LinfPGDAttack(model=model,
                       epsilon=2.0,
                       num_steps=10,
                       step_size=2.0,
                       random_start=1)
output, y = model.logit, model.y_input
loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=y))
with tf.Session(config=gpuconfig) as sess:
    batch_num = int(len(x_nat) / eval_batch_size)
    total_correct = 0
    total_loss = 0
    saver.restore(sess, cur_checkpoint)
    print("restore checkpoint:{}".format(cur_checkpoint))
    for ii in range(batch_num):
        x_batch = x_nat[ii*eval_batch_size:(ii+1)*eval_batch_size, :]
        y_batch = cifar.eval_data.ys[ii*eval_batch_size:(ii+1)*eval_batch_size]
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
        val_dict = {model.x_input: x_batch_adv,
                    model.y_input: y_batch}
        num_correct, batch_loss = sess.run([model.num_correct,loss], feed_dict=val_dict)
        total_correct += num_correct
        total_loss += batch_loss
    print("adv test accuracy:{}".format(total_correct/len(x_nat)))
    print("adv loss:{}".format(total_loss/batch_num))


from PIL import Image
im = Image.fromarray(cifar.eval_data.xs[0, :])
im.save("natural_0.jpeg")
im = Image.fromarray(np.uint8(x_adv[0, :]))
im.save("adv_0.jpeg")
"""
