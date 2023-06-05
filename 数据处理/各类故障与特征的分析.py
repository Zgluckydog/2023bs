# -*- coding = utf-8 -*-
# @Time : 2023/3/14 23:35
# @Author : 张贵
# @File : 各类故障与特征的分析.py
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
import os
# 魔法命令，绘图会好看
# get_ipython().run_line_magic('matplotlib', 'inline')

# 让jupyter中的图画上的中文都显示出来
path ='C:\\Users\\86173\\Desktop'
plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

# <h2>1.调用数据集</h2>

# In[2]:


data_low = pd.read_csv('第二、三问数据集/低级机器设备数据集.csv')
data_med = pd.read_csv('第二、三问数据集/中级机器设备数据集.csv')
data_high = pd.read_csv('第二、三问数据集/高级机器设备数据集.csv')


# In[3]:


def split_dataset(dataset):
    data = dataset.copy()
    data = data[data['具体故障类别'] != 'Normal']
    data.reset_index(drop=True, inplace=True)
    print(data['具体故障类别'].value_counts())
    return data


# <h4>低级</h4>

# In[4]:
data_lows = split_dataset(data_low)
data = data_low.copy()
data = data[data['具体故障类别'] != 'Normal']
data.reset_index(drop=True, inplace=True)
print(data['具体故障类别'].value_counts())
data1 = data['具体故障类别'].value_counts()
data1 = data1.to_frame()
print(type(data1))
if os.access("故障数据/data_lows.xlsx", os.F_OK):
    print ("Given file path is exist.")
    os.remove("故障数据/data_lows.xlsx")
    print("delete sussessful")
outputpath = '故障数据/data_lows.xlsx'
data1.to_excel(outputpath)

# <h4>中级</h4>

# In[5]:


data_meds = split_dataset(data_med)
data_meds
data = data_med.copy()
data = data[data['具体故障类别'] != 'Normal']
data.reset_index(drop=True, inplace=True)
print(data['具体故障类别'].value_counts())
data1 = data['具体故障类别'].value_counts()
data1 = data1.to_frame()
print(type(data1))
if os.access("故障数据/data_meds.xlsx", os.F_OK):
    print ("Given file path is exist.")
    os.remove("故障数据/data_meds.xlsx")
    print("delete sussessful")
outputpath = '故障数据/data_meds.xlsx'
data1.to_excel(outputpath)

# <h4>高级</h4>

# In[6]:


data_highs = split_dataset(data_high)
data_meds
data = data_high.copy()
data = data[data['具体故障类别'] != 'Normal']
data.reset_index(drop=True, inplace=True)
print(data['具体故障类别'].value_counts())
data1 = data['具体故障类别'].value_counts()
data1 = data1.to_frame()
print(type(data1))
if os.access("故障数据/data_highs.xlsx", os.F_OK):
    print ("Given file path is exist.")
    os.remove("故障数据/data_highs.xlsx")
    print("delete sussessful")
outputpath = '故障数据/data_highs.xlsx'
data1.to_excel(outputpath)


# <h2>2.可视化函数</h2>

# <h4>柱图</h4>

# In[7]:


def visul_size(dataset, name, label):
    data = dataset
    bad_type = []
    for i in data['具体故障类别'].value_counts():
        bad_type.append(i)

    # 编辑图例
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # 设置图例字体、位置、数值等等
            plt.text(rect.get_x(), 1.01 * height, '%s' %
                     float(height), size=15, family="Times new roman")

    plt.figure(figsize=(12, 7))
    labels = label
    colors = ['yellow', 'cyan', 'red', 'blue', 'magenta']
    plt.title(str(name) + '故障数量分布图(柱图)', fontsize=20)
    rect = plt.bar(range(len(bad_type)), bad_type, color=colors)
    autolabel(rect)
    plt.xticks(range(len(bad_type)), labels)
    plt.xlabel('故障类型', fontsize=15)
    plt.ylabel('故障数量', fontsize=15)
    plt.show()


# In[8]:


visul_size(data_lows, '低级', label=['OSF', 'HDF', 'PWF', 'TWF', 'RNF'])

# In[9]:


visul_size(data_meds, '中级', label=['HDF', 'PWF', 'TWF', 'OSF', 'RNF'])

# In[10]:


visul_size(data_highs, '高级', label=['HDF', 'TWF', 'PWF', 'OSF'])
# print(data_lows)
# print(data_meds)
# print(data_highs)
# <h4>饼图</h4>

# In[11]:


def visul_pie(dataset, name, name1, labels):
    # 饼图
    data = dataset
    bad_type = []
    for i in data['具体故障类别'].value_counts():
        bad_type.append(i)
    print(bad_type)

    fig = plt.figure(figsize=(12, 7))
    plt.style.use('fivethirtyeight')
    plt.pie(bad_type, labels=labels, autopct='%1.1f%%', counterclock=False, startangle=90)
    plt.title(str(name) + '机器设备故障类别分布图(饼图)')
    plt.show()
    fig.savefig(path + name1, format='svg', dpi=150)  # 输出


# In[36]:


visul_pie(data_lows, '低级', '饼图1.svg',labels=['OSF', 'HDF', 'PWF', 'TWF', 'RNF'])

# In[13]:


visul_pie(data_meds, '中级', '饼图2.svg', labels=['HDF', 'PWF', 'TWF', 'OSF', 'RNF'])

# In[14]:


visul_pie(data_highs, '高级', '饼图3.svg', labels=['HDF', 'TWF', 'PWF', 'OSF'])
print("osf为：")
print(data_lows["具体故障类别"]== "OSF")
# <h2>3.各等级功率分析</h2>

# In[15]:


def p_any(dataset, error_type):
    data = dataset.copy()
    data = data[data['具体故障类别'] == error_type]  # 获取具体类型的错误数据集

    return data


def p_qua(dataset, title):
    # 功率和质量之间的关系
    datas = dataset.copy()
    datas.sort_values(['功率（KW）'], ascending=True, inplace=True)  # 从小到大排序
    datas.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(12, 7))
    plt.title(str(title) + '功率故障关系图', fontsize=20)
    plt.ylim(0, 15)
    plt.scatter(range(len(datas['是否发生故障'])), datas['功率（KW）'])
    plt.ylabel('功率（KW）')
    plt.xlabel('功率从低到高排序')
    plt.show()


# <h4>低级</h4>

# In[16]:


for i in [data_lows]:
    print(i)
    for k in ['OSF', 'HDF', 'PWF', 'TWF', 'RNF']:
        data_ = p_any(i, k)
        p_qua(data_, '低级' + str(k))

    # <h4>中级</h4>

# In[17]:


for i in [data_meds]:
    # print(i)
    for k in ['HDF', 'PWF', 'TWF', 'OSF', 'RNF']:
        data_ = p_any(i, k)
        p_qua(data_, '中级' + str(k))

    # <h4>高级</h4>

# In[18]:


for i in [data_highs]:
    # print(i)
    for k in ['HDF', 'TWF', 'PWF', 'OSF']:
        data_ = p_any(i, k)
        p_qua(data_, '高级' + str(k))

    # <h2>4.各等级时长分析</h2>


# In[19]:


def time_any(dataset, error_type):
    data = dataset.copy()
    data = data[data['具体故障类别'] == error_type]  # 获取具体类型的错误数据集

    return data


def time_qua(dataset, title):
    # 时长和质量之间的关系
    datas = dataset.copy()
    datas.sort_values(['使用时长（min）'], ascending=True, inplace=True)  # 从小到大排序
    datas.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(12, 7))
    plt.title(str(title) + '时长故障关系图', fontsize=20)
    plt.scatter(range(len(datas['是否发生故障'])), datas['使用时长（min）'])
    plt.ylim(0, 250)
    plt.ylabel('使用时长（min）')
    plt.xlabel('使用时长从低到高排序')
    plt.show()


# <h4>低级</h4>

# In[20]:


for i in [data_lows]:
    print(i)
    for k in ['OSF', 'HDF', 'PWF', 'TWF', 'RNF']:
        print(time_any(i, k))
        data_ = time_any(i, k)

        time_qua(data_, '低级' + str(k))

    # <h4>中级</h4>

# In[21]:


for i in [data_meds]:
    # print(i)
    for k in ['HDF', 'PWF', 'TWF', 'OSF', 'RNF']:
        data_ = time_any(i, k)
        time_qua(data_, '中级' + str(k))

    # <h2>5.各等级转速分析</h2>


# In[22]:


def speed_any(dataset, error_type):
    data = dataset.copy()
    data = data[data['具体故障类别'] == error_type]  # 获取具体类型的错误数据集

    return data


def speed_qua(dataset, title):
    # 转速和质量之间的关系
    datas = dataset.copy()
    datas.sort_values(['转速（rpm）'], ascending=True, inplace=True)  # 从小到大排序
    datas.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(12, 7))
    plt.title(str(title) + '转速故障关系图', fontsize=20)
    plt.scatter(range(len(datas['是否发生故障'])), datas['转速（rpm）'])
    # plt.ylim(0,250)
    plt.ylabel('转速（rpm）')
    plt.xlabel('转速从低到高排序')
    plt.show()


# <h4>低级</h4>

# In[23]:


for i in [data_lows]:
    # print(i)
    for k in ['OSF', 'HDF', 'PWF', 'TWF', 'RNF']:
        data_ = speed_any(i, k)
        speed_qua(data_, '低级' + str(k))

    # <h4>中级</h4>

# In[24]:


for i in [data_meds]:
    # print(i)
    for k in ['HDF', 'PWF', 'TWF', 'OSF', 'RNF']:
        data_ = speed_any(i, k)
        speed_qua(data_, '中级' + str(k))

    # <h4>高级</h4>

# In[25]:


for i in [data_highs]:
    # print(i)
    for k in ['HDF', 'TWF', 'PWF', 'OSF']:
        data_ = speed_any(i, k)
        speed_qua(data_, '高级' + str(k))

    # <h2>6.各扭矩分析</h2>


# In[26]:


def distance_any(dataset, error_type):
    data = dataset.copy()
    data = data[data['具体故障类别'] == error_type]  # 获取具体类型的错误数据集

    return data


def distance_qua(dataset, title):
    # 扭矩和质量之间的关系
    datas = dataset.copy()
    datas.sort_values(['扭矩（Nm）'], ascending=True, inplace=True)  # 从小到大排序
    datas.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(12, 7))
    plt.title(str(title) + '扭矩故障关系图', fontsize=20)
    plt.scatter(range(len(datas['是否发生故障'])), datas['扭矩（Nm）'])
    # plt.ylim(0,250)
    plt.ylabel('扭矩（Nm）')
    plt.xlabel('扭矩从低到高排序')
    plt.show()


# <h4>低级</h4>

# In[27]:


for i in [data_lows]:
    # print(i)
    for k in ['OSF', 'HDF', 'PWF', 'TWF', 'RNF']:
        data_ = distance_any(i, k)
        distance_qua(data_, '低级' + str(k))

    # <h4>中级</h4>

# In[28]:


for i in [data_meds]:
    # print(i)
    for k in ['HDF', 'PWF', 'TWF', 'OSF', 'RNF']:
        data_ = distance_any(i, k)
        distance_qua(data_, '中级' + str(k))

    # <h4>高级</h4>

# In[29]:


for i in [data_highs]:
    # print(i)
    for k in ['HDF', 'TWF', 'PWF', 'OSF']:
        data_ = distance_any(i, k)
        distance_qua(data_, '高级' + str(k))

    # <h2>7.厂房室温分析</h2>


# In[30]:


def house_any(dataset, error_type):
    data = dataset.copy()
    data = data[data['具体故障类别'] == error_type]  # 获取具体类型的错误数据集

    return data


def house_qua(dataset, title):
    # 厂房温度和质量之间的关系
    datas = dataset.copy()
    datas.sort_values(['厂房室温（℃）'], ascending=True, inplace=True)  # 从小到大排序
    datas.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(12, 7))
    plt.title(str(title) + '厂房室温故障关系图', fontsize=20)
    plt.scatter(range(len(datas['是否发生故障'])), datas['厂房室温（℃）'])
    # plt.ylim(0,250)
    plt.ylabel('厂房室温（℃）')
    plt.xlabel('厂房室温从低到高排序')
    plt.show()


# <h4>中级</h4>

# In[31]:


for i in [data_meds]:
    # print(i)
    for k in ['HDF', 'PWF', 'TWF', 'OSF', 'RNF']:
        data_ = house_any(i, k)
        house_qua(data_, '中级' + str(k))

    # <h4>高级</h4>

# In[32]:


for i in [data_highs]:
    # print(i)
    for k in ['HDF', 'TWF', 'PWF', 'OSF']:
        data_ = house_any(i, k)
        house_qua(data_, '高级' + str(k))

    # <h2>7.机器温度分析</h2>


# In[33]:


def mech_any(dataset, error_type):
    data = dataset.copy()
    data = data[data['具体故障类别'] == error_type]  # 获取具体类型的错误数据集

    return data


def mech_qua(dataset, title):
    # 机器温度和质量之间的关系
    datas = dataset.copy()
    datas.sort_values(['机器温度（℃）'], ascending=True, inplace=True)  # 从小到大排序
    datas.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(12, 7))
    plt.title(str(title) + '机器温度故障关系图', fontsize=20)
    plt.scatter(range(len(datas['是否发生故障'])), datas['机器温度（℃）'])
    # plt.ylim(0,250)
    plt.ylabel('机器温度（℃）')
    plt.xlabel('机器温度从低到高排序')
    plt.show()


# <h4>中级</h4>

# In[34]:


for i in [data_meds]:
    # print(i)
    for k in ['HDF', 'PWF', 'TWF', 'OSF', 'RNF']:
        data_ = mech_any(i, k)
        mech_qua(data_, '中级' + str(k))

    # <h4>高级</h4>

# In[35]:


for i in [data_highs]:
    # print(i)
    for k in ['HDF', 'TWF', 'PWF', 'OSF']:
        data_ = mech_any(i, k)
        mech_qua(data_, '高级' + str(k))

    # In[ ]:




