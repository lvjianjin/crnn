[English](readme.md) | 简体中文

## 介绍

一个基于Tensorflow2开发的crnn项目。

## 特性

1. 支持数字识别。

## 必要条件

- Python 3.6+
- Tensorflow 2.4.1

## 近期更新

**`2021-2-20`**: 代码重构。

## 内容

- [安装](#安装)
    - [python](#python)
- [使用](#使用)
    - [训练](#训练)
    - [测试](#测试)
    
## 安装

### python

安装所需Python包,
```
pip install -r requirements.txt
```

## 使用

### 训练
1. 拉取项目代码

```
git clone https://github.com/lvjianjin/crnn.git
```
2. 准备数据集

下载数据集至本地，并将本地数据集路径修改至./configs/config.py中的dataset_path。
```
链接：https://pan.baidu.com/s/1FgdITVrM_HsyNh7QSpePjw 
提取码：iakr
解压密码:chineseocr
```
3. 训练

执行训练代码,
```
python main.py -m train
```
### 测试
执行训练代码,
```
python main.py -m test
```
## 联系

1. 邮箱：jianjinlv@163.com
2. QQ群：1081332609

## 许可证书

本项目的代码基于MIT协议发布。