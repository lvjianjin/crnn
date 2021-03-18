# -*- coding: utf-8 -*-
# @Time      : 2021/2/22 17:39
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : test
# @Software  : PyCharm
# @Dscription: 测试文件

import os
import tensorflow as tf
<<<<<<< HEAD
from crnn.data.preprocess.dataset_preprocess import DataGenerator, Dataset
from configs.config import params
from crnn.metrics import Accuracy
from crnn.utils.accuracy import acc
from crnn.losses.layer import CTCLayer
from crnn.data.postprocess.decode import decode_batch_predictions
=======
from configs.config import params
from crnn.data.postprocess.decode import decode_batch_predictions
from crnn.modeling.base_model import BaseModel
from crnn.data.preprocess.dataset_preprocess import DataGenerator, Dataset
>>>>>>> dev


def test(param):
    # 读取图片路径与对应标签
<<<<<<< HEAD
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
=======
    data = Dataset(param, 'test')
    test_image_paths, test_image_labels = data.read()
    # 生成验证集
    test_data_generator = DataGenerator(param=param,
                                         data=test_image_paths,
                                         labels=test_image_labels,
                                         shuffle=False)
    labels_to_char = test_data_generator.labels_to_char
    # 构建模型
    basemodels = BaseModel(param=params)
    model = basemodels.build()
    model.load_weights(os.path.join(param['save_path'], 'crnn_{0}.h5'.format(str(param['test_epoch']))))
    # 构建预测模型
    prediction_model = tf.keras.models.Model(model.get_layer(name='input_data').input,
                                             model.get_layer(name='dense').output)
    prediction_model.summary()
    # 预测
    #  Let's check results on some validation samples
    for p, (inp_value, _) in enumerate(test_data_generator):
        bs = inp_value['input_data'].shape[0]
        X_data = inp_value['input_data']
        labels = inp_value['input_label']

        preds = prediction_model.predict(X_data)
        pred_texts = decode_batch_predictions(preds)

        orig_texts = []
        for label in labels:
            text = ''.join([labels_to_char[int(x)] for x in label if x > 0])
            orig_texts.append(text)

        for i in range(bs):
            print(f'Ground truth: {orig_texts[i]} \t Predicted: {pred_texts[i]}')
        # break
>>>>>>> dev


if __name__ == '__main__':
    test(params)
