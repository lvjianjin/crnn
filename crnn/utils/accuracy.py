# -*- coding: utf-8 -*-
# @Time      : 2021/2/27 17:10
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : accuracy
# @Software  : PyCharm
# @Dscription: 计算准确率

import difflib


def acc(label_list, result_list):
    """
    计算准确率
    严格准确率、相似准确率
    """
    flag_1 = 0
    flag_2 = 0
    for i in range(len(label_list)):
        if label_list[i] == result_list[i]:
            flag_1 += 1
        flag_2 += difflib.SequenceMatcher(None, label_list[i], result_list[i]).quick_ratio()
    return flag_1/len(label_list), flag_2/len(label_list)

