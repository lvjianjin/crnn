# -*- coding: utf-8 -*-
# @Time      : 2021/2/22 9:38
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : dataset_read
# @Software  : PyCharm
# @Dscription: 读取数据集

import os
import cv2
import pickle
import random
from pathlib import Path


class Dataset:
    """
    读取数据集
    """

    def __init__(self, param):
        self.path = param["dataset_path"]
        self.index_path = param["dataset_index_path"]
        self.radio = param["input_features"][1] / param["input_features"][0]

    def read(self, mode="train"):
        """
        读取索引文件，若读取不到就去创建该文件再读取
        :param mode: 模式；有train和test两种模式
        :return: 返回图片路径及对应标签
        """
        dataset_path = os.path.join(self.index_path, mode, 'dataset.data')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                train_image_paths, train_image_labels, val_image_paths, val_image_labels = pickle.load(f)
        else:
            train_image_paths, train_image_labels, val_image_paths, val_image_labels = self.search(dataset_path)
        print('数据集读取完毕！')
        return train_image_paths, train_image_labels, val_image_paths, val_image_labels

    def search(self, path):
        print('开始获取数据集，请耐心等待...')
        images = []
        train_image_paths = []
        train_image_labels = []
        val_image_paths = []
        val_image_labels = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if '.jpg' in file:
                    file_path = os.path.join(root, file)
                    label_path = file_path.replace('.jpg', '.txt')
                    if Path(file_path.replace('.jpg', '.txt')).exists():
                        with open(label_path) as f:
                            label = f.read().strip()
                        imgs = cv2.imread(file_path)
                        if imgs.shape[1] / imgs.shape[0] <= self.radio and len(label) > 0:
                            images.append((file_path, label))
        random.shuffle(images)
        for image, label in images:
            random_num = random.randint(1, 20)
            if random_num == 5:
                val_image_paths.append(image)
                val_image_labels.append(label)
            else:
                train_image_paths.append(image)
                train_image_labels.append(label)
        with open(path, 'wb') as f:
            pickle.dump((train_image_paths, train_image_labels, val_image_paths, val_image_labels), f)
        return train_image_paths, train_image_labels, val_image_paths, val_image_labels


if __name__ == '__main__':
    from configs.config import params

    data = Dataset(params)
    train_image_paths, train_image_labels, val_image_paths, val_image_labels = data.read()