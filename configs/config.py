# -*- coding: utf-8 -*-
# @Time      : 2021/2/20 17:23
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : config
# @Software  : PyCharm
# @Dscription: 配置文件

# 字符类型当前支持三种
# chinese ==> 中文
# digit   ==> 数字
# date    ==> 日期
char_type = 'chinese'

params = {
    # 是否重新训练模型
    "retrain": False,
    # 训练轮数
    "epochs": 300,
    # 训练初始轮数
    "initial_epoch": 4,
    # 初始学习率
    "initial_learning_rate": 0.0001,
    # 预处理方法
    "preprocess": "keras",
    # 批次大小
    "batch": 16,
    # 打乱数据规模
    "buffer": 1000,
    # 最大标签长度
    "max_length": 20,
    # 下采样次数
    "downsample_factor": 5,
    # 日志路径
    "log_dir": './logs/{0}'.format(char_type),
    # 字典路径
    "table_path": "./configs/tables/{0}/ch_old.txt".format(char_type),
    # cnn骨干网络
    "cnn_model": 'vgg',
    # rnn骨干网络
    "rnn_model": 'lstm',
    # 输入图片大小
    "input_features": [32, 320, 3],
    # 训练集路径
    "train_dataset_path": "/data/dataset/ocr/crnn/{0}/train".format(char_type),
    # 验证集路径
    "val_dataset_path": "/data/dataset/ocr/crnn/{0}/val".format(char_type),
    # 测试数据路径
    "test_dataset_path": r"D:\dataset\ocr\crnn\chinese\test\true\cropped_imgs",
    # 数据集索引路径
    "dataset_index_path": "./datasets",
    # 模型保存路径
    "save_path": "./outputs/{0}".format(char_type),
    # 准确率方法
    "accuracy": "SequenceAccuracy",
    # 测试模型序号
    "test_epoch": 30,
    "transform_model_save_path": "./deploy/multiModel/{0}/2".format(char_type),
}
