# -*- coding: utf-8 -*-
# @Time      : 2021/2/20 17:39
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : lstm
# @Software  : PyCharm
# @Dscription: 

import tensorflow as tf


def lstm(x):
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=256, return_sequences=True, use_bias=True, recurrent_activation='sigmoid'))(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=256, return_sequences=True, use_bias=True, recurrent_activation='sigmoid'))(x)
    return x
