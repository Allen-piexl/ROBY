# -*- coding: utf-8 -*-
"""
@author: Zhen-Wang
训练鲁棒模型
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from model_lib import MobileNet
from pgd_attack import LinfPGDAttack
from tensorflow.examples.tutorials.mnist import input_data

data_path = '/public/home/cjy/Documents/Dataset/DNN/fashion_data'
model_dir = '/public/home/cjy/Documents/Python/DNN/code-mnist/mobilenet_eval/models-f/robust_mobilenet'
tf.set_random_seed(12345)
np.random.seed(54321)

mnist = input_data.read_data_sets(data_path, one_hot=False)
batch_size = 50
eval_batch_size = 200
max_num_training_steps = 80000
num_output_steps = 100
num_summary_steps = 100
num_checkpoint_steps = 300
model = MobileNet(is_training=True)
global_step = tf.contrib.framework.get_or_create_global_step()

output, y = model.logit, model.y_input

loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=y))
#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

attack = LinfPGDAttack(model = model,
                         epsilon = 0.3,    #cifar10 为1.0
                         num_steps = 10,
                         step_size = 2.0,
                         random_start = 1)

# Setting up the Tensorboard and checkpoint outputs

if not os.path.exists(model_dir):
  os.makedirs(model_dir)

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('xent adv train', loss)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

os.environ['CUDA_VISIBLE_DEVICES']='4'
gpuconfig = tf.ConfigProto()
#gpuconfig.gpu_options.per_process_gpu_memory_fraction = 0.8 # 占用GPU80%的显存
gpuconfig.gpu_options.allow_growth = True
with tf.Session(config=gpuconfig) as sess:
    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    training_time = 0.0
    val_acc_pre = 0.0
    min_loss = 10.0
    # Main training loop
    for ii in range(max_num_training_steps):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        # Compute Adversarial Perturbations
        start = timer()
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
        end = timer()
        training_time += end - start

        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch}

        adv_dict = {model.x_input: x_batch_adv,
                    model.y_input: y_batch}

        # Output to stdout
        if ii % num_output_steps == 0:
            nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
            los = sess.run(loss, feed_dict=adv_dict)
            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
            print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
            print('loss : {}'.format(los))
            if los < min_loss:
                min_loss = los
                print('minloss : {}'.format(min_loss))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                training_time = 0.0

        # Tensorboard summaries
        if ii % num_summary_steps == 0:
            summary = sess.run(merged_summaries, feed_dict=adv_dict)
            summary_writer.add_summary(summary, ii)

        # Write a checkpoint
        # 测试评估一下
        if ii % num_checkpoint_steps == 0:
            x_batch = mnist.test.images[0:eval_batch_size, :]
            y_batch = mnist.test.labels[0:eval_batch_size]
            x_batch_adv = attack.perturb(x_batch, y_batch, sess)
            val_dict = {model.x_input: x_batch_adv,
                        model.y_input: y_batch}
            val_acc = sess.run(model.accuracy, feed_dict=val_dict)
            print('Valuation accuracy {:.4}%'.format(val_acc * 100))
            if val_acc > val_acc_pre:
                saver.save(sess,
                           os.path.join(model_dir, 'checkpoint'),
                           global_step=ii)
                val_acc_pre = val_acc

        # Actual training step
        start = timer()
        sess.run(train_step, feed_dict=adv_dict)
        end = timer()
        training_time += end - start

