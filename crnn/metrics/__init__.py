# -*- coding: utf-8 -*-
# @Time      : 2021/2/22 17:17
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : __init__
# @Software  : PyCharm
# @Dscription: 准确率指标

from crnn.metrics.sequenceAcc import SequenceAccuracy
from crnn.metrics.editDistanceAcc import EditDistance
from configs.config import params

if params['accuracy'] == 'EditDistance':
    Accuracy = EditDistance
else:
    Accuracy = SequenceAccuracy
