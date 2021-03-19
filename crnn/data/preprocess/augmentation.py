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
        self.w = weight
        self.h = hight

    def blur(self, image):  # 均值滤波模糊
        """
        滤波
        """
        rand = random.randint(1, 3)
        if rand == 1:
            # 均值滤波
            kenerl = random.randint(1, 2) * 2 + 1
            return cv2.blur(image, (kenerl, kenerl))
        elif rand == 2:
            # 高斯滤波
            kenerl = random.randint(1, 2) * 2 + 1
            return cv2.GaussianBlur(image, (kenerl, kenerl), 0, 0)
        else:
            # 中值滤波
            kenerl = random.randint(1, 2) * 2 + 1
            return cv2.medianBlur(image, kenerl)

    def random_brightness(self, image):
        """
        明度变化
        """
        c = random.uniform(0.5, 3.5)
        blank = np.zeros(image.shape, image.dtype)
        dst = cv2.addWeighted(image, c, blank, 1 - c, 0)
        return dst

    def rotate(self, image, scale=1.0):
        """
        旋转
        """
        angle = random.uniform(-3, 3)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def saltAndPepper(self, image):
        """
        椒盐噪声
        """
        percetage = random.uniform(0., 0.1)
        SP_NoiseImg = image.copy()
        SP_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
        for i in range(SP_NoiseNum):
            randR = np.random.randint(0, image.shape[0] - 1)
            randG = np.random.randint(0, image.shape[1] - 1)
            randB = np.random.randint(0, 3)
            if np.random.randint(0, 1) == 0:
                SP_NoiseImg[randR, randG, randB] = 0
            else:
                SP_NoiseImg[randR, randG, randB] = 255
        return SP_NoiseImg

    def addGaussianNoise(self, image):
        """
        高斯噪声
        """
        percetage = random.uniform(0., 0.1)
        G_Noiseimg = image.copy()
        w = image.shape[1]
        h = image.shape[0]
        G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
        for i in range(G_NoiseNum):
            temp_x = np.random.randint(0, h)
            temp_y = np.random.randint(0, w)
            G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
        return G_Noiseimg

    def adjust_hue(self, image):
        """
        调整图片的色度
        添加到色调通道的量在-1和1之间的间隔。
        如果值超过180，则会旋转这些值。
        """
        if random.randint(1, 10) > 8:
            delta = random.uniform(-0.05, 0.05)
            image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)  # 取余数
        return image

    def adjust_saturation(self, image):
        """
        调整图片的饱和度
        """
        factor = random.uniform(0.95, 1.05)
        image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
        return image

    def apply(self, image):
        # 滤波
        image = self.blur(image)
        # 亮度
        image = self.random_brightness(image)
        # 旋转
        image = self.rotate(image)
        # 椒盐噪声
        image = self.saltAndPepper(image)
        # 高斯噪声
        image = self.addGaussianNoise(image)
        # 色度
        image = self.adjust_hue(image)
        # 饱和度
        image = self.adjust_saturation(image)
        return image


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # 原图
    path = r"D:\dataset\ocr\crnn\bank_card\test\0.jpg"
    img = cv2.imread(path)
    print(img.shape)
    h, w, _ = img.shape
    plt.imshow(img)
    # 图像增强的类
    augment = Augment(h, w)
    # # 色彩反转
    img1 = augment.apply(img)
    plt.imshow(img1)

    print(img1.shape)