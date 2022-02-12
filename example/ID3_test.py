# -*- coding:utf-8 -*-
# @Time : 2022/2/12 4:59 下午
# @Author : huichuan LI
# @File : ID3_test.py
# @Software: PyCharm
from class_model.ID3 import ID3
import numpy as np
import pandas as pd
from math import log

df = pd.read_csv('../data/example_data.csv', dtype={'windy': 'str'})

tree1 = ID3(df, 'play')
tree1.construct_tree()
tree1.print_tree(tree1.root, "")
