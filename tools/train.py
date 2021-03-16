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
    # 读取图片路径与对应标签
    data = Dataset(param)
    train_image_paths, train_image_labels, val_image_paths, val_image_labels = data.read()
    # 生成训练集
    train_data_generator = DataGenerator(param=params,
                                         data=train_image_paths,
                                         labels=train_image_labels,
                                         shuffle=True)
    # 生成验证集
    valid_data_generator = DataGenerator(param=params,
                                         data=val_image_paths,
                                         labels=val_image_labels,
                                         shuffle=False)
    # 构建模型
    basemodels = BaseModel()
    if param['retrain']:
        model = basemodels.build()
        # 模型编译
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=param["initial_learning_rate"]),
            metrics=['accuracy']
        )
    else:
        # 自定义层
        _custom_objects = {
                            "CTCLayer": CTCLayer,
                          }
        model = tf.keras.models.load_model(
            os.path.join(
                param['save_path'],
                'crnn_{0}.h5'.format(str(param['initial_epoch']))
            ),
            custom_objects=_custom_objects)
    # 回调函数
    callbacks = [
        # 模型保存
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(param['save_path'], "crnn_{epoch}.h5"),
            monitor='val_loss',
            verbose=1
        ),
        # 提前结束规则
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=5,
                                         restore_best_weights=True)
    ]
    # 查看模型结构
    model.summary()
    # 模型训练
    model.fit(
        train_data_generator,
        validation_data=valid_data_generator,
        epochs=params["epochs"],
        callbacks=callbacks
    )


if __name__ == '__main__':
    train(params)
