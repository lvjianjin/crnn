# -*- coding: utf-8 -*-
# @Time      : 2021/2/21 22:28
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : ctc_loss
# @Software  : PyCharm
# @Dscription: ctc损失函数

import tensorflow as tf


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, logits_time_major=False, blank_index=-1,
                 reduction=tf.keras.losses.Reduction.AUTO, name='ctc_loss'):
        super().__init__(reduction=reduction, name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index
        )
        return tf.reduce_mean(loss)

