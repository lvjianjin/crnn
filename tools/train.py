# -*- coding: utf-8 -*-
# @Time      : 2021/2/20 18:23
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : train
# @Software  : PyCharm
# @Dscription: 模型训练

from crnn.data.preprocess.dataset_preprocess import Preprocess
from crnn.modeling.base_model import BaseModel
from configs.config import params
import tensorflow as tf
import datetime
import os


def train(param):
    # 构建训练集
    dataset = Preprocess(param)
    ds_train = dataset.build('train')
    ds_train_size = dataset.size('train')
    ds_val = dataset.build('val')
    ds_val_size = dataset.size('val')
    # 构建模型
    basemodels = BaseModel(param=param)
    if param['retrain']:
        model = basemodels.build()
    else:
        model = basemodels.build()
        model.load_weights(os.path.join(param['save_path'], 'crnn_{0}.h5'.format(str(param['initial_epoch']))))
    # 回调函数
    callbacks = [
        # 模型保存
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(param['save_path'], "crnn_{epoch}.h5"),
            monitor='val_loss',
            save_weights_only=True,
            verbose=1
        ),
        # 提前结束规则
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=20,
                                         restore_best_weights=True),
        # tensorboard
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(param['log_dir'],
                                                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
                                       histogram_freq=1),
        # 学习率自动下降机制
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001)
    ]
    # 查看模型结构
    model.summary()
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
    train(params)
