# -*- coding: utf-8 -*-
# @Time      : 2021/2/20 16:46
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : vggnet
# @Software  : PyCharm
# @Dscription: VggNet骨干网络

import tensorflow as tf


def vgg(x):
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-05, axis=1, momentum=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(0, 1))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-05, axis=1, momentum=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(0, 1))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=2, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-05, axis=1, momentum=0.1)(x)
    x = tf.keras.layers.Reshape((-1, 512))(x)
    return x


if __name__ == '__main__':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    input = Input(shape=(32, 320, 3), name='the_input')
    basemodel = Model(inputs=input, outputs=vgg(input))
    basemodel.summary()
