# -*- coding: utf-8 -*-
# @Time      : 2021/3/3 16:34
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : layer
# @Software  : PyCharm
# @Dscription: CTC层

import tensorflow as tf


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        # On test time, just return the computed loss
        return loss

    @classmethod
    def from_config(cls, config):
        return cls(**config)
