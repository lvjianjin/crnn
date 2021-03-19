# -*- coding: utf-8 -*-
# @Time      : 2021/2/20 17:39
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : lstm
# @Software  : PyCharm
# @Dscription: RNN块

import tensorflow as tf


def lstm(x, output_features):
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=256, return_sequences=True, use_bias=True, recurrent_activation='sigmoid'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256))(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=256, return_sequences=True, use_bias=True, recurrent_activation='sigmoid'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_features, activation='softmax'))(x)
    return x
