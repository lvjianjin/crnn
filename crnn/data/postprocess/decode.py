# -*- coding: utf-8 -*-
# @Time      : 2021/2/25 17:26
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : decode
# @Software  : PyCharm
# @Dscription: 结果解析

import tensorflow as tf


class Decoder:
    def __init__(self, param, merge_repeated=True):
        """
        Args:
            table: list, char map
            blank_index: int(default: NUM_CLASSES - 1), the index of blank
        label.
            merge_repeated: bool
        """
        with open(param["table_path"], 'r', encoding='gbk') as f:
            self.table = [char.strip() for char in f]
            blank_index = len(self.table)
        self.blank_index = blank_index
        self.merge_repeated = merge_repeated

    def map2string(self, inputs):
        strings = []
        for i in inputs:
            text = [self.table[char_index] for char_index in i
                    if char_index != self.blank_index]
            strings.append(''.join(text))
        return strings

    def decode(self, inputs, from_pred=True, method='greedy'):
        if from_pred:
            logit_length = tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1])
            if method == 'greedy':
                decoded, _ = tf.nn.ctc_greedy_decoder(
                    inputs=tf.transpose(inputs, perm=[1, 0, 2]),
                    sequence_length=logit_length,
                    merge_repeated=self.merge_repeated)
            elif method == 'beam_search':
                decoded, _ = tf.nn.ctc_beam_search_decoder(
                    inputs=tf.transpose(inputs, perm=[1, 0, 2]),
                    sequence_length=logit_length)
            inputs = decoded[0]
        decoded = tf.sparse.to_dense(inputs,
                                     default_value=self.blank_index).numpy()
        decoded = self.map2string(decoded)
        return decoded