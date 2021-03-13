# -*- coding: utf-8 -*-
# @Time      : 2021/2/20 17:23
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : config
# @Software  : PyCharm
# @Dscription: 配置文件

params = {
    # 是否重新训练模型
    "retrain": True,
    # 训练轮数
    "epochs": 30,
    # 训练初始轮数
    "initial_epoch": 0,
    # 初始学习率
    "initial_learning_rate": 0.001,
    # 批次大小
    "batch": 256,
    # 打乱数据规模
    "buffer": 10000,
    # 最大标签长度
    "max_length": 20,
    # 下采样次数
    "downsample_factor": 4,
    # 字典路径
    "table_path": "./configs/table.txt",
    # cnn骨干网络
    "cnn_model": 'vgg',
    # rnn骨干网络
    "rnn_model": 'lstm',
    # 输入图片大小
    "input_features": [32, 480, 3],
    # 训练集路径
    "dataset_path": r"D:\dataset\train\ocr\crnn\bank_card",
    # 测试数据路径
    "test_path": "./datasets/test",
    # 数据集索引路径
    "dataset_index_path": "./datasets",
    # 模型保存路径
    "save_path": "./outputs",
    # 准确率方法
    "accuracy": "SequenceAccuracy",
    # 测试模型序号
    "test_epoch": 7,
}
