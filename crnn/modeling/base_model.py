# -*- coding: utf-8 -*-
# @Time      : 2021/2/20 17:17
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : base_model
# @Software  : PyCharm
# @Dscription: 基础网络结构


import tensorflow as tf
from configs.config import params
from crnn.modeling.backbone.vggnet import vgg
from crnn.modeling.backbone.lstm import lstm


class BaseModel:
    """
    基础网络结构
    """
    def __init__(self, cnn_network=vgg, rnn_network=lstm, param=params):
        self.input_features = (param['input_features'][0], None, param['input_features'][2])
        self.cnn_network = cnn_network
        self.rnn_network = rnn_network
        self.dense = tf.keras.layers.Dense(units=param['output_features'])

    def build(self):
        inputs = tf.keras.Input(shape=self.input_features)
        x = self.cnn_network(inputs)
        x = self.rnn_network(x)
        outputs = self.dense(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="crnn")

