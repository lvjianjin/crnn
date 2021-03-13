# -*- coding: utf-8 -*-
# @Time      : 2021/2/22 9:27
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : dataset_preprocess
# @Software  : PyCharm
# @Dscription: 数据集预处理

import math
import cv2
import numpy as np
import tensorflow as tf
from configs.config import params
from crnn.data.preprocess.dataset_read import Dataset


class DataGenerator(tf.keras.utils.Sequence):
    """Generates batches from a given dataset.

    Args:
        data: training or validation data
        labels: corresponding labels
        char_map: dictionary mapping char to labels
        batch_size: size of a single batch
        img_width: width of the resized
        img_height: height of the resized
        downsample_factor: by what factor did the CNN downsample the images
        max_length: maximum length of any captcha
        shuffle: whether to shuffle data or not after each epoch
    Returns:
        batch_inputs: a dictionary containing batch inputs
        batch_labels: a batch of corresponding labels
    """

    def __init__(self,
                 param,
                 data,
                 labels,
                 shuffle=True
                 ):
        self.data = data
        self.labels = labels
        self.batch_size = param['batch']
        self.img_width = param['input_features'][1]
        self.img_height = param['input_features'][0]
        self.img_color = param['input_features'][2]
        self.downsample_factor = param['downsample_factor']
        self.max_length = param['max_length']
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.on_epoch_end()
        with open(param['table_path'], 'r') as f:
            lines = f.readlines()
            self.char_map = dict(zip([line.replace('\n', '') for line in lines], list(range(len(lines)))))

    def is_valid_captcha(self, captcha):
        """
        Sanity check for corrupted images
        """
        for ch in captcha:
            if not ch in self.char_map:
                return False
        return True

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        # 1. Get the next batch indices
        curr_batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # 2. This isn't necessary but it can help us save some memory
        # as not all batches the last batch may not have elements
        # equal to the batch_size
        batch_len = len(curr_batch_idx)

        # 3. Instantiate batch arrays
        batch_images = np.ones((batch_len, self.img_height, self.img_width, self.img_color),
                               dtype=np.float32)
        batch_labels = np.ones((batch_len, self.max_length), dtype=np.float32)
        input_length = np.ones((batch_len, 1), dtype=np.int64) * \
                       (self.img_width // self.downsample_factor - 2)
        label_length = np.zeros((batch_len, 1), dtype=np.int64)

        for j, idx in enumerate(curr_batch_idx):
            # 1. Get the image and transpose it
            img = cv2.imread(self.data[idx])
            # resize图片
            h = img.shape[0]
            w = img.shape[1]
            ratio = w / float(h)
            if math.ceil(self.img_height * ratio) > self.img_width:
                resized_w = self.img_width
            else:
                resized_w = int(math.ceil(self.img_height * ratio))
            resized_image = cv2.resize(img, (resized_w, self.img_height))
            resized_image = resized_image.astype('float32')
            padding_im = np.zeros((self.img_height, self.img_width, self.img_color), dtype=np.float32)
            padding_im[:, 0:resized_w, :] = resized_image
            # 3. Get the correpsonding label
            text = self.labels[idx]
            # padding label
            padding_label = np.ones((self.max_length,), dtype=np.int64)*(-1)
            padding_label[0:len(text)] = [self.char_map[ch] for ch in text]
            # 4. Include the pair only if the captcha is valid
            if self.is_valid_captcha(text):
                label = padding_label
                batch_images[j] = padding_im
                batch_labels[j] = label
                label_length[j] = len(text)

        batch_inputs = {
            'input_data': batch_images,
            'input_label': batch_labels,
            'input_length': input_length,
            'label_length': label_length,
        }
        return batch_inputs, np.zeros(batch_len).astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == '__main__':
    # 字典
    with open(params['table_path'], 'r') as f:
        lines = f.readlines()
        char_to_labels = dict(zip([line.replace('\n', '') for line in lines], list(range(len(lines)))))
    # 读取图片路径与对应标签
    data = Dataset(params)
    train_image_paths, train_image_labels, val_image_paths, val_image_labels = data.read()
    # 生成数据集
    train_data_generator = DataGenerator(param=params,
                                         data=train_image_paths,
                                         labels=train_image_labels,
                                         shuffle=True)
    valid_data_generator = DataGenerator(param=params,
                                         data=val_image_paths,
                                         labels=val_image_labels,
                                         shuffle=False)
    print(train_data_generator)
