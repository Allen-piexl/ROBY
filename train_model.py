# -*- coding: utf-8 -*-
"""
@author: Zhen-Wang
"""
#import cifar10_input
import model_lib as lib
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
from timeit import default_timer as timer
from tensorflow.examples.tutorials.mnist import input_data


data_path = '/public/home/cjy/Documents/Dataset/DNN/fashion_data'
model_dir = '/public/home/cjy/Documents/Python/DNN/code-mnist/mobilenet_eval/models-f/natural_mobilenet'
batch_size = 50
eval_batch_size = 200
max_num_training_steps = 100000
num_output_steps = 100
num_summary_steps = 100
num_checkpoint_steps = 300
model = lib.MobileNet(is_training=True)
saver = tf.train.Saver(max_to_keep=3)

tf.set_random_seed(4557077)
np.random.seed(54321)

output, y = model.logit, model.y_input

loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=y))
#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.allow_growth = True

tf.summary.scalar('accuracy  train', model.accuracy)
tf.summary.scalar('loss train', loss)
tf.summary.image('images train', model.x_image)
merged_summaries = tf.summary.merge_all()

mnist = input_data.read_data_sets(data_path, one_hot=False)
with tf.Session(config=gpuconfig) as sess:


    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    training_time = 0.0
    val_acc_pre = 0.0
    min_loss = 10.0
    for ii in range(max_num_training_steps):

        x_batch, y_batch = mnist.train.next_batch(batch_size)
        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch}

        if ii % num_output_steps == 0:
            nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            los = sess.run(loss, feed_dict=nat_dict)
            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
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
            summary = sess.run(merged_summaries, feed_dict=nat_dict)
            summary_writer.add_summary(summary, ii)

        # 测试评估一下
        if ii % num_checkpoint_steps == 0:
            x_batch = mnist.test.images[0:eval_batch_size, :]
            y_batch = mnist.test.labels[0:eval_batch_size]
            val_dict = {model.x_input: x_batch,
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
        sess.run(train_step, feed_dict=nat_dict)
        end = timer()
        training_time += end - start

