#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DCGAN.py
# Author: Yuxin Wu

import argparse
import glob
import numpy as np
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.viz import stack_patches

from GAN import GANModelDesc, GANTrainer, RandomZData

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


"""
1. Download the 'aligned&cropped' version of CelebA dataset
   from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

2. Start training:
    ./DCGAN-CelebA.py --data /path/to/img_align_celeba/ --crop-size 140
    Generated samples will be available through tensorboard

3. Visualize samples with an existing model:
    ./DCGAN-CelebA.py --load path/to/model --sample

You can also train on other images (just use any directory of jpg files in
`--data`). But you may need to change the preprocessing.

A pretrained model on CelebA is at http://models.tensorpack.com/#GAN
"""


class Model(GANModelDesc):
    def __init__(self, shape, batch, z_dim):
        self.shape = 512
        self.batch = batch
        self.zdim = z_dim

    def inputs(self):
        return [tf.TensorSpec((None, self.shape, self.shape, 3), tf.float32, 'input')]

    def generator(self, z):
        nf = 16
        l = FullyConnected('fc0', z, nf * 64 * 4 * 4, nl=tf.identity)
        l = tf.reshape(l, [-1, 4, 4, nf * 64])
        l = BNReLU(l)
        with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
            l = Deconv2D('deconv1', l, [8, 8, nf * 32])
            l = Deconv2D('deconv2', l, [16, 16, nf * 16])
            l = Deconv2D('deconv3', l, [32, 32, nf*8])
            l = Deconv2D('deconv4', l, [64, 64, nf * 4])
            l = Deconv2D('deconv5', l, [128, 128, nf * 2])
            l = Deconv2D('deconv6', l, [256, 256, nf * 1])
            l = Deconv2D('deconv7', l, [512, 512, 3], nl=tf.identity)
            l = tf.tanh(l, name='gen')
        return l

    @auto_reuse_variable_scope
    def discriminator(self, imgs):
        nf = 16
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
                argscope(LeakyReLU, alpha=0.2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf, nl=LeakyReLU)
                 .Conv2D('conv1', nf * 2)
                 .BatchNorm('bn1').LeakyReLU()
                 .Conv2D('conv2', nf * 4)
                 .BatchNorm('bn2').LeakyReLU()
                 .Conv2D('conv3', nf * 8)
                 .BatchNorm('bn3').LeakyReLU()
                 .Conv2D('conv4', nf * 16)
                 .BatchNorm('bn4').LeakyReLU()
                 .Conv2D('conv5', nf * 32)
                 .BatchNorm('bn5').LeakyReLU()
                 .Conv2D('conv6', nf * 64)
                 .BatchNorm('bn6').LeakyReLU()
                 .FullyConnected('fct', 1, nl=tf.identity)())
        return l
    
    def build_graph(self, inputs):
        image_pos = inputs[0]
        image_pos = image_pos / 128.0 - 1

        z = tf.random.uniform([self.batch, self.zdim], -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, self.zdim], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.generator(z)
            tf.summary.image('generated-samples', image_gen, max_outputs=30)
            with tf.variable_scope('discrim'):
                vecpos = self.discriminator(image_pos)
                vecneg = self.discriminator(image_gen)

        self.build_losses(vecpos, vecneg)
        self.collect_variables()

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def get_augmentors():
    augs = []
    if args.load_size:
        augs.append(imgaug.Resize(args.load_size))
    if args.crop_size:
        augs.append(imgaug.CenterCrop(args.crop_size))
    augs.append(imgaug.Resize(args.final_size))
    return augs


def get_data():
    assert args.data
    imgs = glob.glob(args.data + '/*.png')
    print(imgs)
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = AugmentImageComponent(ds, get_augmentors())
    ds = BatchData(ds, args.batch)
    ds = MultiProcessRunnerZMQ(ds, 5)
    return ds


def sample(model, model_path, output_name='gen/gen'):
    pred = PredictConfig(
        session_init=SmartInit(model_path),
        model=model,
        input_names=['z'],
        output_names=[output_name, 'z'])
    pred = SimpleDatasetPredictor(pred, RandomZData((100, args.z_dim)))
    for o in pred.get_result():
        o = o[0] + 1
        o = o * 128.0
        o = np.clip(o, 0, 255)
        o = o[:, :, :, ::-1]
        stack_patches(o, nr_row=10, nr_col=10, viz=True)


def get_args(default_batch=128, default_z_dim=100):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='view generated examples')
    parser.add_argument('--data', help='a jpeg directory')
    parser.add_argument('--load-size', help='size to load the original images', type=int)
    parser.add_argument('--crop-size', help='crop the original images', type=int)
    parser.add_argument(
        '--final-size', default=64, type=int,
        help='resize to this shape as inputs to network')
    parser.add_argument('--z-dim', help='hidden dimension', type=int, default=default_z_dim)
    parser.add_argument('--batch', help='batch size', type=int, default=default_batch)
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


if __name__ == '__main__':
    args = get_args()
    M = Model(shape=args.final_size, batch=args.batch, z_dim=args.z_dim)
    if args.sample:
        sample(M, args.load)
    else:
        logger.auto_set_dir()
        GANTrainer(
            input=QueueInput(get_data()),
            model=M).train_with_defaults(
            callbacks=[ModelSaver()],
            steps_per_epoch=300,
            max_epoch=200,
            session_init=SmartInit(args.load),
        )
