# -*- coding: utf-8 -*-
# @Time      : 2021/2/22 17:39
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : test
# @Software  : PyCharm
# @Dscription: 测试文件

import os
import tensorflow as tf
from configs.config import params
from crnn.data.postprocess.decode import decode_batch_predictions
from crnn.modeling.base_model import BaseModel
from crnn.data.preprocess.dataset_preprocess import DataGenerator, Dataset


def test(param):
    # 读取图片路径与对应标签
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


if __name__ == '__main__':
    test(params)
