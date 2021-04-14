# -*- coding: utf-8 -*-
# @Time      : 2021/2/22 17:39
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : test
# @Software  : PyCharm
# @Dscription: 测试文件

import os
from crnn.data.preprocess.dataset_preprocess import Preprocess
from crnn.data.postprocess.decode import Decoder
from configs.config import params
from crnn.modeling.base_model import BaseModel
from crnn.utils.accuracy import acc


def test(param):
    dataset = Preprocess(param)
    test_data_generator, test_labels = dataset.build_test()
    # 构建模型
    basemodels = BaseModel(param=params)
    model = basemodels.build()
    model.load_weights(os.path.join(param['save_path'], 'crnn_{0}.h5'.format(str(param['test_epoch']))))
    model.summary()
    # 预测
    test_data = next(iter(test_data_generator))
    result = model.predict(test_data)

    decoder = Decoder(param)
    y_pred = decoder.decode(result, method='greedy')
    for i, sentense in enumerate(y_pred):
        if test_labels[i] != sentense:
            print('真实标签：{0} \t 预测结果： {1}'.format(test_labels[i], sentense))

    # 准确率
    acc1, acc2 = acc(y_pred, test_labels)
    print("========================================")
    print("严格准确率：{0} \t 相似准确率：{1}".format(acc1, acc2))


if __name__ == '__main__':
    test(params)
