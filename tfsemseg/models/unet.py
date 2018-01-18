import tensorflow as tf

from tfsemseg.models.utils import *

def unet(x, n_classes=21, feature_scale=4):
    filters = [64, 128, 256, 512, 1024]
    filters = [(f/feature_scale) for f in filters]
    kernel_size = (3, 3)
    pool_size = (2, 2)

    # downsample
    conv1 = tf.layers.conv2d(inputs=x, filters=filters[0], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv1 = tf.layers.conv2d(inputs=conv1, filters=filters[0], kernel_size=kernel_size, padding="same",activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool_size, strides=pool_size)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=filters[1], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv2, filters=filters[1], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=pool_size, strides=pool_size)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=filters[2], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv3, filters=filters[2], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=pool_size, strides=pool_size)    

    conv4 = tf.layers.conv2d(inputs=pool3, filters=filters[3], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv4, filters=filters[3], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=pool_size, strides=pool_size)

    conv5 = tf.layers.conv2d(inputs=pool4, filters=filters[4], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(inputs=conv5, filters=filters[4], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)

    # upsample
    up6 = UpSampling2D(conv5, pool_size)
    up6 = tf.concat([up6, conv4], axis=3)
    conv6 = tf.layers.conv2d(inputs=up6, filters=filters[3], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(inputs=conv6, filters=filters[3], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
      
    up7 = UpSampling2D(conv6, pool_size)
    up7 = tf.concat([up7, conv3], axis=3)
    conv7 = tf.layers.conv2d(inputs=up7, filters=filters[2], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv7 = tf.layers.conv2d(inputs=conv7, filters=filters[2], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)

    up8 = UpSampling2D(conv7, pool_size)
    up8 = tf.concat([up8, conv2], axis=3)
    conv8 = tf.layers.conv2d(inputs=up8, filters=filters[1], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv8 = tf.layers.conv2d(inputs=conv8, filters=filters[1], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    
    up9 = UpSampling2D(conv8, pool_size)
    up9 = tf.concat([up9, conv1], axis=3)
    conv9 = tf.layers.conv2d(inputs=up9, filters=filters[0], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    conv9 = tf.layers.conv2d(inputs=conv9, filters=filters[0], kernel_size=kernel_size, padding="same", activation=tf.nn.relu)

    out = tf.layers.conv2d(inputs=conv9, filters=n_classes, kernel_size=(1, 1))
    
    return out
