#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DCGAN.py
# Author: Yuxin Wu

import argparse
import glob
import numpy as np
import os
import tensorflow as tf
import timeit
import imageio
import skimage
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
    def get_inputs(self):
        return [InputDesc(tf.float32, (None, 256, 256, 3), 'input')]

    def __init__(self, shape, batch, z_dim):
        self.shape = shape
        self.batch = batch
        self.zdim = z_dim

    def inputs(self):
        return [tf.TensorSpec((None, self.shape, self.shape, 3), tf.float32, 'input')]


    # def generator(self, z):
    #     nf = 16
    #     l = FullyConnected('fc0', z, nf * 64 * 4 * 4, nl=tf.identity)
    #     l = tf.reshape(l, [-1, 4, 4, nf * 64])
    #     l = BNReLU(l)
    #     with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
    #         l = Deconv2D('deconv1', l, [8, 8, nf * 32])
    #         l = Deconv2D('deconv2', l, [16, 16, nf * 16])
    #         l = Deconv2D('deconv3', l, [32, 32, nf*8])
    #         l = Deconv2D('deconv4', l, [64, 64, nf * 4])
    #         l = Deconv2D('deconv5', l, [128, 128, nf * 2])
    #         l = Deconv2D('deconv6', l, [256, 256, nf * 1])
    #         l = Deconv2D('deconv7', l, [512, 512, 3], nl=tf.identity)
    #         l = tf.tanh(l, name='gen')
    #     return l

    @auto_reuse_variable_scope
    def discriminator(self, imgs):
        nf = 16
        imgs = tf.reshape(imgs, [-1, 256, 256, 3])
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
                 .FullyConnected('fct', 1, nl=tf.identity)())
        return l
        
    def generator(self, z):
        """ return an image generated from z"""
        nf = 16
        l = FullyConnected('fc0', z, nf * 64 * 4 * 4, activation=tf.identity)
        l = tf.reshape(l, [-1, 4, 4, nf * 64])
        l = BNReLU(l)
        with argscope(Conv2DTranspose, activation=BNReLU, kernel_size=4, strides=2):
            l = Conv2DTranspose('deconv2', l, nf * 16)
            l = Conv2DTranspose('deconv3', l, nf * 8)
            l = Conv2DTranspose('deconv4', l, nf * 4)
            l = Conv2DTranspose('deconv5', l, nf * 2)
            l = Conv2DTranspose('deconv6', l, nf)
            l = Conv2DTranspose('deconv7', l, 3, activation=tf.identity)
            l = tf.tanh(l, name='gen')
        return l

    # @auto_reuse_variable_scope
    # def discriminator(self, imgs):
    #     nf = 16
    #     with argscope(Conv2D, kernel_size=4, strides=2):
    #         l = (LinearWrap(imgs)
    #             .Conv2D('conv0', nf, activation=tf.nn.leaky_relu)
    #             .Conv2D('conv1', nf * 2)
    #             .BatchNorm('bn1')
    #             .tf.nn.leaky_relu()
    #             .Conv2D('conv2', nf * 4)
    #             .BatchNorm('bn2')
    #             .tf.nn.leaky_relu()
    #             .Conv2D('conv3', nf * 8)
    #             .BatchNorm('bn3')
    #             .tf.nn.leaky_relu()
    #             .Conv2D('conv4', nf * 16)
    #             .BatchNorm('bn4')
    #             .tf.nn.leaky_relu()
    #             .Conv2D('conv5', nf * 32)
    #             .BatchNorm('bn5')
    #             .tf.nn.leaky_relu()
    #             .Conv2D('conv6', nf * 64)
    #             .BatchNorm('bn6')
    #             .tf.nn.leaky_relu()
    #             .FullyConnected('fct', 1)())
    #     return l

    def build_graph(self, image_pos):
        image_pos = image_pos / 128.0 - 1

        z = tf.random.uniform([self.batch, self.zdim], -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, self.zdim], name='z')

        with argscope([Conv2D, Conv2DTranspose, FullyConnected],
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
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
    i = 0
    for o in pred.get_result():
        o = o[0] + 1
        o = o * 128.0
        o = np.clip(o, 0, 255)
        o = o[:, :, :, ::-1]

        imageio.imwrite("output/"+str(i)+".png", o[i])
        i+=1


def get_args(default_batch=32, default_z_dim=512):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='view generated examples')
    parser.add_argument('--data', help='a jpeg directory')
    parser.add_argument('--load-size', help='size to load the original images', type=int)
    parser.add_argument('--crop-size', help='crop the original images', type=int)
    parser.add_argument(
        '--final-size', default=256, type=int,
        help='resize to this shape as inputs to network')
    parser.add_argument('--z-dim', help='hidden dimension', type=int, default=default_z_dim)
    parser.add_argument('--batch', help='batch size', type=int, default=default_batch)
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args

def sample2(model, model_path,sample_path, num, output_name='gen/gen'):
    config = PredictConfig(
        session_init=SmartInit(model_path),
        model=model,
        input_names=['z'],
        output_names=[output_name, 'z'])
    graph = config._maybe_create_graph()
    batch_size = 250
    n = 0
    with graph.as_default():
        input = PlaceholderInput()
        input.setup(config.model.get_inputs_desc())
        with TowerContext('', is_training=False):
            config.model.build_graph(input)
        input_tensors = get_tensors_by_names(config.input_names)
        output_tensors = get_tensors_by_names(config.output_names)
        sess = config.session_creator.create_session()
        config.session_init.init(sess)
        if sess is None:
            sess = tf.get_default_session()
        start = timeit.default_timer()
        if (num % batch_size != 0):
            num_extra_img = num % batch_size
            dp = [np.random.normal(-1, 1, size=(num_extra_img, opt.Z_DIM))]
            feed = dict(zip(input_tensors, dp))
            output = sess.run(output_tensors, feed_dict=feed)
            o, zs = output[0] + 1, output[1]
            for j in range(len(o)):
                n = n + 1
                img = o[j]
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                img = (img - np.min(img))/np.ptp(img)
                imageio.imwrite('%s%09d.jpeg' % (sample_path,n), skimage.img_as_ubyte(img))
        for i in  range(int(num/batch_size)):
            dp = [np.random.normal(-1, 1, size=(batch_size, opt.Z_DIM))]
            feed = dict(zip(input_tensors, dp))
            output = sess.run(output_tensors, feed_dict=feed)
            o, zs = output[0] + 1, output[1]
            for j in range(len(o)):
                n = n + 1
                img = o[j]
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                img = (img - np.min(img))/np.ptp(img)
                imageio.imwrite('%s%09d.jpeg' % (sample_path,n), skimage.img_as_ubyte(img))
        print ("Images generated : ", str(num))
        stop = timeit.default_timer()
        print ("Time taken : ", str(stop - start))


if __name__ == '__main__':
    args = get_args()
    M = Model(shape=args.final_size, batch=32, z_dim=512)
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
