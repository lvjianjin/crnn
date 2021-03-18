# -*- coding: utf-8 -*-
# @Time      : 2021/3/18 14:28
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : augmentation
# @Software  : PyCharm
# @Dscription: 图像增强方法

import cv2
import numpy as np
import random


class Augment:
    def __init__(self, weight, hight):
        self.inv_prob = 0.5
        self.blur_prob = 0.3
        self.sq_blur_prob = 0.3
        self.bright_prob = 0.5
        self.rotate_prob = 1.
        self.zoom_prob = 1.
        self.gray_prob = 0.0
        self.w = weight
        self.h = hight

    def invert(self, image):  # 色彩反转
        return 255 - image

    def blur(self, image):  # 均值滤波模糊
        return cv2.blur(image, (3, 3))

    def sq_blur(self, image):  # 区域插值模糊，模拟图像低分辨率
        image = cv2.resize(image, (self.h, self.w), interpolation=cv2.INTER_AREA)
        return image

    def random_brightness(self, image):  # 明度变化
        c = random.uniform(0.2, 1.8)
        blank = np.zeros(image.shape, image.dtype)
        dst = cv2.addWeighted(image, c, blank, 1 - c, 0)
        return dst

    def rotate(self, image, scale=1.0):  # 旋转
        angle = random.uniform(-5, 5)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def zoom(self, image, scale=0.3):  # 缩放（根据个人需要）
        h, w = image.shape[:2]
        w_ = int(w * self.h / h)
        if w_ > self.w:
            return image
        else:
            w_ = random.randint(max(1, int(w_ * (1 - scale))), w_)
            image = cv2.resize(image, (w_, self.h))
            return image

    def gray_scale(self, image):  # 灰度化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.merge((gray, gray, gray))
        return dst

    def apply(self, image):
        inv_prob = random.random()
        blur_prob = random.random()
        sq_blur_prob = random.random()
        bright_prob = random.random()
        rotate_prob = random.random()
        zoom_prob = random.random()
        if inv_prob < self.inv_prob:
            image = self.invert(image)
        if bright_prob < self.bright_prob:
            image = self.random_brightness(image)
        if rotate_prob < self.rotate_prob:
            image = self.rotate(image)
        if zoom_prob < self.zoom_prob:
            image = self.zoom(image)
        if blur_prob < self.blur_prob:
            image = self.blur(image)
        if sq_blur_prob < self.sq_blur_prob:
            image = self.sq_blur(image)
        return image
