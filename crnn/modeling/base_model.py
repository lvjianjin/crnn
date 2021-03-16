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
from crnn.losses.layer import CTCLayer


class BaseModel:
    """
    基础网络结构
    """
    def __init__(self, rnn_network=lstm, param=params):
        with open(param['table_path'], 'r') as f:
            output_features = len(f.readlines()) + 1
        self.input_features = (param['input_features'][0], None, param['input_features'][2])
        self.rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        if param['cnn_model'] == 'dense':
            self.cnn_network = dense
        else:
            self.cnn_network = vgg
        self.rnn_network = rnn_network
        self.dense = tf.keras.layers.Dense(output_features, activation='softmax')

    def build(self):
        # 输入层
        inputs_img = tf.keras.Input(shape=self.input_features, name='input_data')
        labels = tf.keras.layers.Input(name='input_label', shape=[20], dtype='float32')
        input_length = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')
        label_length = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')
        # 图片标准化
        x = self.rescaling(inputs_img)
        # cnn结构网络层
        x = self.cnn_network(x)
        # rnn结构网络层
        x = self.rnn_network(x)
        # 输出层
        x = self.dense(x)
        # Loss层
        outputs = CTCLayer(name='ctc_loss')(labels, x, input_length, label_length)
        return tf.keras.Model(inputs=[inputs_img, labels, input_length, label_length], outputs=outputs, name="crnn")

