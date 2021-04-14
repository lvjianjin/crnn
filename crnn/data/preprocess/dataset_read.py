# -*- coding: utf-8 -*-
# @Time      : 2021/2/22 9:38
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : dataset_read
# @Software  : PyCharm
# @Dscription: 读取数据集

import os
import cv2
import glob
import pickle
import random
from pathlib import Path
import concurrent.futures


class Dataset:
    """
    读取数据集
    """

    def __init__(self, param, mode='train'):
        self.mode = mode
        if self.mode == 'train':
            self.path = param["train_dataset_path"]
        elif self.mode == 'val':
            self.path = param["val_dataset_path"]
        else:
            self.path = param["test_dataset_path"]
        self.index_path = param["dataset_index_path"]
        self.radio = param["input_features"][1] / param["input_features"][0]
        self.max_length = param['max_length']
        with open(param['table_path'], 'r', encoding='utf8') as f:
            lines = f.readlines()
            self.char_map = dict(zip([line.replace('\n', '') for line in lines], list(range(len(lines)))))
        self.images = []

    def is_valid_captcha(self, captcha):
        """
        Sanity check for corrupted images
        """
        for ch in captcha:
            if not ch in self.char_map:
                return False
        return True

    def read(self):
        """
        读取索引文件，若读取不到就去创建该文件再读取
        :param mode: 模式；有train和test两种模式
        :return: 返回图片路径及对应标签
        """
        dataset_path = os.path.join(self.index_path, self.mode, 'dataset.data')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                train_image_paths, train_image_labels = pickle.load(f)
        else:
            train_path = os.path.join(self.index_path, self.mode)
            if not os.path.exists(train_path):
                os.mkdir(train_path)
            train_image_paths, train_image_labels = self.search(dataset_path)
        print('数据集读取完毕！')
        return train_image_paths, train_image_labels

    def preprocess(self, file_path):
        label_path = file_path.replace('.jpg', '.txt')
        if Path(file_path.replace('.jpg', '.txt')).exists():
            with open(label_path) as f:
                label = f.read().strip()
            imgs = cv2.imread(file_path)
            if imgs.shape[1] / imgs.shape[0] <= self.radio and len(label) > 0 and len(
                    label) <= self.max_length and self.is_valid_captcha(label):
                self.images.append((file_path, label))

    def search(self, path):
        file_paths = []
        print('开始获取数据集，请耐心等待...')
        train_image_paths = []
        train_image_labels = []
        print('开始获取图片路径列表...')
        # 获取所有图片路径
        for root, dirs, files in os.walk(self.path):
            file_paths += glob.glob(os.path.join(root, '*.jpg'))
        print('图片路径列表获取完毕！')
        print('正在处理图片...')
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            executor.map(self.preprocess, file_paths)
        random.shuffle(self.images)
        for image, label in self.images:
            train_image_paths.append(image)
            train_image_labels.append(label)
        with open(path, 'wb') as f:
            pickle.dump((train_image_paths, train_image_labels), f)
        return train_image_paths, train_image_labels


if __name__ == '__main__':
    from configs.config import params

    data = Dataset(params, 'val')
    train_image_paths, train_image_labels = data.read()
