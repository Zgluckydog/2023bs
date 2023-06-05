# -*- coding = utf-8 -*-
# @Time : 2023/3/15 8:58
# @Author : 张贵
# @File : test1.py
# @Software : PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minepy import MINE
from sklearn.preprocessing import OrdinalEncoder  # 把特征转换成分类数值
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler  # 归一化
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sqlite3
import xlwt

# 魔法命令，绘图会好看
# get_ipython().run_line_magic('matplotlib', 'inline')

# 让jupyter中的图画上的中文都显示出来

plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

# <h2>1.调用数据集</h2>

# In[2]:


data_low = pd.read_csv(r'C:\Users\86173\Desktop\毕设\数据处理\第二、三问数据集\低级机器设备数据集.csv')
data_med = pd.read_csv(r'C:\Users\86173\Desktop\毕设\数据处理\第二、三问数据集\中级机器设备数据集.csv')
data_high = pd.read_csv(r'C:\Users\86173\Desktop\毕设\数据处理\第二、三问数据集\高级机器设备数据集.csv')


# In[3]:


def split_dataset(dataset):
    data = dataset.copy()
    data = data[data['具体故障类别'] != 'Normal']
    data.reset_index(drop=True, inplace=True)
    # print(data['具体故障类别'].value_counts())

    return data['具体故障类别'].value_counts()


# <h4>低级</h4>

# In[4]:
savepath = "data_lows.xls"
data_lows = []
data_lows = split_dataset(data_low)
# print(data_lows)
book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet(savepath, cell_overwrite_ok=True)
col = ( "故障类型", "数量")
for i in range(0, 2):
    sheet.write(0, i, col[i])
for j in range(0, 5):
    print("第%d条" % (j + 1))
    data = data_lows[j]

book.save(savepath)  # 保存

conn = sqlite3.connect("data_lows.db")  # 打开或创建数据库文件
print("成功打开数据库")

c = conn.cursor()  # 获取游标

sql = '''
    create table IF NOT EXISTS data_lows
    (
        ltype text,
        num numeric
    );
    '''

c.execute(sql)  # 执行sql语句
conn.commit()  # 提交数据库操作
conn.close()  # 关闭数据库链接

conn = sqlite3.connect("data_lows.db")
cur = conn.cursor()
for data in data_lows:
    for index in range(len(data)):
        data[index] = '"' + data[index] + '"'
        sql = '''
                insert into data_lows (
                ltype,num)
                '''
    cur.execute(sql)
    conn.commit()
cur.close()
conn.close()