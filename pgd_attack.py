# -*- coding: utf-8 -*-
"""
@author: Zhen-Wang
此代码的作用为对训练好的原始模型进行样本的PGD攻击，生成对抗样本。

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

#import cifar10_input

class LinfPGDAttack:
    def __init__(self, model, epsilon, num_steps, step_size, random_start):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start

        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.logit, labels=model.y_input))
        # 出来的是一个列表，取第一个元素为代表x_input导数的数组
        self.grad = tf.gradients(loss, model.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, 255)  # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        for i in range(self.num_steps):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.model.y_input: y})

            x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 255)  # ensure valid pixel range

        return x

if __name__ == '__main__':
  import sys
  import math

  from model_lib import MobileNet

  #model_dir = "models/natural_alexnet"
  #model_dir = "models/robust_alexnet"
  data_path = '/public/home/cjy/Documents/Dataset/DNN/fashion_data'
  model_dir = '/public/home/cjy/Documents/Python/DNN/code-mnist/mobilenet_eval/models-f/robust_mobilenet'

  model_file = tf.train.latest_checkpoint(model_dir)
  if model_file is None:
    print('No model found')
    sys.exit()

  model = MobileNet(is_training=True)
  attack = LinfPGDAttack(model = model,
                         epsilon = 0.3,   #cifar10 为1.0
                         num_steps = 10,
                         step_size = 2.0,
                         random_start = 1)
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets(data_path, one_hot=False)

  os.environ['CUDA_VISIBLE_DEVICES'] = '3'
  gpuconfig = tf.ConfigProto()
  gpuconfig.gpu_options.allow_growth = True

  with tf.Session(config=gpuconfig) as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)
    print("restore checkpoint:{}".format(model_file))

    # Iterate over the samples batch-by-batch
    num_eval_examples = 10000
    eval_batch_size = 200
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = "/public/home/cjy/Documents/Python/DNN/code-mnist/mobilenet_eval/fashion-robust-attack.npy"
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))