# -*- coding: utf-8 -*-
# @Time      : 2021/2/22 17:16
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : editDistanceAcc
# @Software  : PyCharm
# @Dscription: 

import tensorflow as tf
from tensorflow import keras


class EditDistance(keras.metrics.Metric):
    def __init__(self, name='edit_distance', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float32)
        self.sum_distance = self.add_weight(name='sum_distance',
                                            initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_shape = tf.shape(y_pred)
        batch_size = y_pred_shape[0]
        logit_length = tf.fill([batch_size], y_pred_shape[1])
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        sum_distance = tf.math.reduce_sum(tf.edit_distance(decoded[0], y_true))
        batch_size = tf.cast(batch_size, tf.float32)
        self.sum_distance.assign_add(sum_distance)
        self.total.assign_add(batch_size)

    def result(self):
        return self.sum_distance / tf.clip_by_value(self.total, 1e-8, tf.reduce_max(self.total))

    def reset_states(self):
        self.sum_distance.assign(0)
        self.total.assign(0)
