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
from crnn.modeling.backbone.densenet import dense
from crnn.modeling.backbone.lstm import lstm


class BaseModel:
    """
    基础网络结构
    """
    def __init__(self, rnn_network=lstm, param=params):
        with open(param['table_path'],'r') as f:
            output_features = len(f.readlines()) + 1
        self.input_features = (param['input_features'][0], None, param['input_features'][2])
        self.rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        if param['cnn_model'] == 'dense':
            self.cnn_network = dense
        else:
            self.cnn_network = vgg
        self.rnn_network = rnn_network
        self.dense = tf.keras.layers.Dense(output_features)

    def build(self):
        # 输入层
        inputs = tf.keras.Input(shape=self.input_features)
        # 图片标准化
        x = self.rescaling(inputs)
        # cnn结构网络层
        x = self.cnn_network(x)
        # rnn结构网络层
        x = self.rnn_network(x)
        # 输出层
        outputs = self.dense(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="crnn")

