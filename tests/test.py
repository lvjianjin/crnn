# -*- coding: utf-8 -*-
# @Time      : 2021/2/22 17:39
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : test
# @Software  : PyCharm
# @Dscription: 测试文件

import os
import tensorflow as tf
from crnn.data.preprocess.dataset_preprocess import DataGenerator, Dataset
from configs.config import params
from crnn.metrics import Accuracy
from crnn.utils.accuracy import acc
from crnn.losses.layer import CTCLayer
from crnn.data.postprocess.decode import decode_batch_predictions


def test(param):
    # 读取图片路径与对应标签
    data = Dataset(param)
    train_image_paths, train_image_labels, val_image_paths, val_image_labels = data.read()
    # 生成验证集
    valid_data_generator = DataGenerator(param=params,
                                         data=val_image_paths,
                                         labels=val_image_labels,
                                         shuffle=False)
    # 加载模型
    _custom_objects = {
        "CTCLayer": CTCLayer,
    }
    model_path = os.path.join(param['save_path'], "crnn_{0}.h5".format(str(param["test_epoch"])))
    model = tf.keras.models.load_model(model_path, custom_objects=_custom_objects)
    # 编译模型
    prediction_model = tf.keras.models.Model(model.get_layer(name='input_data').input,
                                             model.get_layer(name='dense2').output)
    # 查看模型结构
    prediction_model.summary()
    # 预测
    test_ds = next(iter(ds))
    result = model.predict(test_ds)
    decoder = Decoder(param)
    y_pred = decoder.decode(result, method='greedy')
    acc_1, acc_2 = acc(labels, y_pred)
    print("Label: ", labels)
    print("Predict: ", y_pred)
    print("WordAcc: ", acc_1)
    print("CharAcc: ", acc_2)


if __name__ == '__main__':
    test(params)
