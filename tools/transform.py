# -*- coding: utf-8 -*-
# @Time      : 2021/3/23 10:48
# @Author    : JianjinL
# @eMail     : jianjinlv@163.com
# @File      : transform
# @Software  : PyCharm
# @Dscription: 

import os
from configs.config import params
from crnn.modeling.base_model import BaseModel

# 构建模型
basemodels = BaseModel(param=params)
model = basemodels.build()
model.load_weights(os.path.join(params['save_path'], 'crnn_{0}.h5'.format(str(params['test_epoch']))))
model.summary()
# 模型保存
model.save(params["transform_model_save_path"])
print('保存成功。')
