# -*- coding = utf-8 -*-
# @Time : 2023/3/15 11:04
# @Author : 张贵
# @File : test2.py
# @Software : PyCharm

import sqlite3
import pandas as pd


df = pd.read_excel('data_low.xlsx')

print(df)

print("数据的总长度为：",len(df),"\n")

for i in range(len(df)):
    print(i)