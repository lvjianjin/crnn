# -*- coding: utf-8 -*-
# @Time      : 2021/2/20 18:23
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : train
# @Software  : PyCharm
# @Dscription: 模型训练

from crnn.data.preprocess.dataset_preprocess import DataGenerator, Dataset
from crnn.modeling.base_model import BaseModel
from crnn.losses.layer import CTCLayer
from configs.config import params
import tensorflow as tf
import os


def train(param):
    # 生成训练集
    train_data = Dataset(param, 'train')
    train_image_paths, train_image_labels = train_data.read()
    train_data_generator = DataGenerator(param=params,
                                         data=train_image_paths,
                                         labels=train_image_labels,
                                         shuffle=True)
    # 生成验证集
    train_data = Dataset(param, 'val')
    val_image_paths, val_image_labels = train_data.read()
    valid_data_generator = DataGenerator(param=params,
                                         data=val_image_paths,
                                         labels=val_image_labels,
                                         shuffle=False)
    # 构建模型
    basemodels = BaseModel(param=params)
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
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                         patience=5,
                                         restore_best_weights=True)
    ]
    # 查看模型结构
    model.summary()
    # 模型训练
    model.fit(
        train_data_generator,
        validation_data=valid_data_generator,
        initial_epoch=param['initial_epoch'],
        epochs=param["epochs"],
        callbacks=callbacks
    )


if __name__ == '__main__':
    train(params)
