# -*- coding: utf-8 -*-
# @Time      : 2021/3/2 15:38
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : densenet
# @Software  : PyCharm
# @Dscription: DenseNet骨干网络

import tensorflow as tf


def conv_block(x, growth_rate, dropout_rate=None):
    """
    cnn块
    """
    x = tf.keras.layers.BatchNormalization(axis=1, epsilon=1.1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2):
    """
    dense块
    """
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(x, nb_filter, dropout_rate=None, pooltype=1):
    """
    transition块
    """
    x = tf.keras.layers.BatchNormalization(axis=1, epsilon=1.1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same')(x)

    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if pooltype == 2:
        x = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif pooltype == 1:
        x = tf.keras.layers.ZeroPadding2D(padding=(0, 1))(x)
        x = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif pooltype == 3:
        x = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter


def dense(x):
    """
    densenet网络
    """
    # 参数
    _dropout_rate = 0.2
    _weight_decay = 1e-4
    _nb_filter = 64
    # conv 64 5*5 s=2
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2)
    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2)
    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None)
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2)
    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None)
    x = tf.keras.layers.BatchNormalization(axis=1, epsilon=1.1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Reshape((-1, 192))(x)
    return x


if __name__ == '__main__':
    input = tf.keras.layers.Input(shape=(32, 480, 3), name='the_input')
    basemodel = tf.keras.models.Model(inputs=input, outputs=dense(input))
    basemodel.summary()
