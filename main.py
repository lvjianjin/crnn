# -*- coding: utf-8 -*-
# @Time      : 2021/2/28 17:24
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : main
# @Software  : PyCharm
# @Dscription: 项目主程序


import sys, getopt
from tools.train import train
from tests.test import test
from configs.config import params


def main(param):
    # 解析命令行参数
    mode = None
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "m:")  # 短选项模式
    except Exception as err:
        print("Error:", err)

    for opt, arg in opts:
        if opt in ['-m']:
            mode = arg
    # 执行训练或测试
    if mode == 'test':
        test(param)
    else:
        train(param)


if __name__ == '__main__':
    main(params)
