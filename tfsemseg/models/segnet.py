import numpy as np
import tensorflow as tf

from tfsemseg.models.utils import *

def segnet(x, n_classes=21, feature_scale=4):

    pool_size = (2, 2)
    kernel_size = (3, 3)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    filters = [64, 128, 256, 512, 512]
    filters = [(f/feature_scale) for f in filters]

    conv1 = tf.layers.conv2d(inputs=x, filters=filters[0], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv1 = tf.layers.conv2d(inputs=conv1, filters=filters[0], kernel_size=kernel_size, padding="same",activation=tf.nn.relu)
    pool1, arg1 = tf.nn.max_pool_with_argmax(input=conv1, ksize=ksize, strides=strides, padding='SAME')

    conv2 = tf.layers.conv2d(inputs=pool1, filters=filters[1], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv2, filters=filters[1], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    pool2, arg2 = tf.nn.max_pool_with_argmax(input=conv2, ksize=ksize, strides=strides, padding='SAME')

    conv3 = tf.layers.conv2d(inputs=pool2, filters=filters[2], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv3, filters=filters[2], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv3, filters=filters[2], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    pool3, arg3 = tf.nn.max_pool_with_argmax(input=conv3, ksize=ksize, strides=strides, padding='SAME')

    conv4 = tf.layers.conv2d(inputs=pool3, filters=filters[3], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv4, filters=filters[3], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv4, filters=filters[3], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    pool4, arg4 = tf.nn.max_pool_with_argmax(input=conv4, ksize=ksize, strides=strides, padding='SAME')

    conv5 = tf.layers.conv2d(inputs=pool4, filters=filters[4], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(inputs=conv5, filters=filters[4], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(inputs=conv5, filters=filters[4], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    pool5, arg5 = tf.nn.max_pool_with_argmax(input=conv5, ksize=ksize, strides=strides, padding='SAME')


    up6 = unpool_with_argmax(pool5, arg5, name='maxunpool5')
    conv6 = tf.layers.conv2d(inputs=up6, filters=filters[4], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(inputs=conv6, filters=filters[4], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(inputs=conv6, filters=filters[4], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)

    up7 = unpool_with_argmax(conv6, arg4, name='maxunpool4')
    conv7 = tf.layers.conv2d(inputs=up7, filters=filters[3], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv7 = tf.layers.conv2d(inputs=conv7, filters=filters[3], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv7 = tf.layers.conv2d(inputs=conv7, filters=filters[3], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)

    up8 = unpool_with_argmax(conv7, arg3, name='maxunpool3')
    conv8 = tf.layers.conv2d(inputs=up8, filters=filters[2], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv8 = tf.layers.conv2d(inputs=conv8, filters=filters[2], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv8 = tf.layers.conv2d(inputs=conv8, filters=filters[2], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)

    up9 = unpool_with_argmax(conv8, arg2, name='maxunpool2')
    conv9 = tf.layers.conv2d(inputs=up9, filters=filters[1], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv9 = tf.layers.conv2d(inputs=conv9, filters=filters[1], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)

    up10 = unpool_with_argmax(conv9, arg1, name='maxunpool1')
    conv10 = tf.layers.conv2d(inputs=up10, filters=filters[0], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv10 = tf.layers.conv2d(inputs=conv10, filters=filters[0], kernel_size=kernel_size, padding="same", activation=None)

    out = tf.layers.conv2d(inputs=conv10, filters=n_classes, kernel_size=(1, 1))
    
    return out
