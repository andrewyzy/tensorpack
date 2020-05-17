#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Improved-WGAN.py
# Author: Yuxin Wu

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils import get_tf_version_tuple
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary

import DCGAN
from GAN import SeparateGANTrainer

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

"""
Improved Wasserstein-GAN.
See the docstring in DCGAN.py for usage.
"""

# Don't want to mix two examples together, but want to reuse the code.
# So here just import stuff from DCGAN.



class Model(DCGAN.Model):
    # replace BatchNorm by LayerNorm
    @auto_reuse_variable_scope
    def discriminator(self, imgs):
        nf = 16
        with argscope(Conv2D, activation=tf.identity, kernel_size=4, strides=2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf, activation=tf.nn.leaky_relu)
                 .Conv2D('conv1', nf * 2)
                 .LayerNorm('ln1')
                 .tf.nn.leaky_relu()
                 .Conv2D('conv2', nf * 4)
                 .LayerNorm('ln2')
                 .tf.nn.leaky_relu()
                 .Conv2D('conv3', nf * 8)
                 .LayerNorm('ln3')
                 .tf.nn.leaky_relu()
                 .Conv2D('conv4', nf * 16)
                 .LayerNorm('ln4')
                 .tf.nn.leaky_relu()
                 .Conv2D('conv5', nf * 32)
                 .LayerNorm('ln5')
                 .tf.nn.leaky_relu()
                 .Conv2D('conv6', nf * 64)
                 .LayerNorm('ln6')
                 .tf.nn.leaky_relu()
                 .FullyConnected('fct', 1, activation=tf.identity)())
        return tf.reshape(l, [-1])

    

    def build_graph(self, image_pos):
        image_pos = image_pos / 128.0 - 1

        z = tf.random_normal([32, 512], name='z_train')
        z = tf.placeholder_with_default(z, [None, 512], name='z')

        with argscope([Conv2D, Conv2DTranspose, FullyConnected],
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.generator(z)
            tf.summary.image('generated-samples', image_gen, max_outputs=30)

            alpha = tf.random_uniform(shape=[32, 1, 1, 1],
                                      minval=0., maxval=1., name='alpha')
            interp = image_pos + alpha * (image_gen - image_pos)

            with tf.variable_scope('discrim'):
                vecpos = self.discriminator(image_pos)
                vecneg = self.discriminator(image_gen)
                vec_interp = self.discriminator(interp)

        # the Wasserstein-GAN losses
        self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
        self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')

        # the gradient penalty loss
        gradients = tf.gradients(vec_interp, [interp])[0]
        gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
        gradients_rms = tf.sqrt(tf.reduce_mean(tf.square(gradients)), name='gradient_rms')
        gradient_penalty = tf.reduce_mean(tf.square(gradients - 1), name='gradient_penalty')
        add_moving_summary(self.d_loss, self.g_loss, gradient_penalty, gradients_rms)

        self.d_loss = tf.add(self.d_loss, 10 * gradient_penalty)

        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
        return opt


if __name__ == '__main__':
    assert get_tf_version_tuple() >= (1, 4)
    args = DCGAN.get_args(default_batch=32, default_z_dim=512)
    M = Model(shape=args.final_size, batch=32, z_dim=512)
    if args.sample:
        DCGAN.sample(M, args.load)
    else:
        logger.auto_set_dir()
        SeparateGANTrainer(
            QueueInput(DCGAN.get_data()),
            M, g_period=5).train_with_defaults(
            callbacks=[ModelSaver()],
            steps_per_epoch=300,
            max_epoch=200,
            session_init=SmartInit(args.load)
        )