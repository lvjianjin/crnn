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


def main():
    # 构建训练集
    dataset = Preprocess(params)
    ds_train = dataset.build('train')
    ds_train_size = dataset.size('train')
    ds_val = dataset.build('val')
    ds_val_size = dataset.size('val')
    # 构建模型
    basemodels = BaseModel()
    model = basemodels.build()
    # 模型编译
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss=CTCLoss(),
        # metrics=[Accuracy()]
    )
    # 回调函数
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(params['save_path'], "crnn_{epoch}.h5"),
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
        steps_per_epoch=ds_train_size // params["batch"],
        initial_epoch=params["initial_epoch"],
        validation_data=ds_val,
        validation_steps=ds_val_size // params["batch"],
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main()
