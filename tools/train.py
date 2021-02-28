# -*- coding: utf-8 -*-
# @Time      : 2021/2/20 18:23
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : train
# @Software  : PyCharm
# @Dscription: 模型训练

from crnn.data.preprocess.dataset_preprocess import Preprocess
from crnn.modeling.base_model import BaseModel
from crnn.metrics import Accuracy
from crnn.losses.ctc_loss import CTCLoss
from configs.config import params
import tensorflow as tf
import os


def train(param):
    # 构建训练集
    dataset = Preprocess(param)
    ds_train = dataset.build('train')
    ds_train_size = dataset.size('train')
    ds_val = dataset.build('val')
    ds_val_size = dataset.size('val')
    # 构建模型
    basemodels = BaseModel()
    if param['retrain']:
        model = basemodels.build()
    else:
        model = tf.keras.models.load_model(
            os.path.join(
                param['save_path'],
                'crnn_{0}.h5'.format(str(param['initial_epoch']))
            ), compile=False)
    # 加载预训练模型

    # 模型编译
    model.compile(
        optimizer=tf.keras.optimizers.Adam(param["learning_rate"]),
        loss=CTCLoss(),
        metrics=[Accuracy()]
    )
    # 回调函数
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(param['save_path'], "crnn_{epoch}.h5"),
            monitor='val_loss',
            verbose=1
        ),
    ]
    # 查看模型结构
    model.summary()
    # 模型训练
    model.fit(
        ds_train,
        epochs=params["epochs"],
        steps_per_epoch=ds_train_size // param["batch"],
        initial_epoch=param["initial_epoch"],
        validation_data=ds_val,
        validation_steps=ds_val_size // param["batch"],
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main(params)
