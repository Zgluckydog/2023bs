# -*- coding = utf-8 -*-
# @Time : 2023/4/2 21:34
# @Author : 张贵
# @File : 预测结果.py.py
# @Software : PyCharm

# !/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 魔法命令，绘图会好看
# get_ipython().run_line_magic('matplotlib', 'inline')

# 让jupyter中的图画上的中文都显示出来

plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

# <h2>1.调用数据集</h2>

# In[2]:


data_lgb = pd.read_csv('第四题lgb修正故障后的预测结果.csv')
data_svm = pd.read_csv('第四题SVM修正故障后的预测结果.csv')
data_rf = pd.read_csv('第四题RF修正故障后的预测结果.csv')
data = pd.read_excel('原数据/forecast.xlsx')

# In[3]:


data_rf['具体故障类别'].value_counts()

# In[4]:


data_lgb['具体故障类别'].value_counts()

# In[5]:


data_svm['具体故障类别'].value_counts()

# <h2>2.投票判决</h2>

# <h4>二分类判决</h4>

# In[6]:


ans = pd.DataFrame()

# In[7]:


ans['2分类判决'] = data_rf['是否发生故障'] + data_lgb['是否发生故障'] + data_svm['是否发生故障']

# In[8]:


set_1 = ans[(ans['2分类判决'] >= 2)].index

# In[9]:


set_0 = ans[(ans['2分类判决'] < 2)].index

# In[10]:


ans.iloc[set_0] = 0
ans.iloc[set_1] = 1

# In[11]:


ans.value_counts()

# In[12]:


ans.columns = ['是否发生故障']

# In[13]:


ans

# In[14]:


ans.value_counts()

# In[15]:


data_rf

# <h4>多分类判决</h4>

# In[16]:


index1 = []
index1_label = []
index2 = []
index2_label = []
for i in range(0, 1000):
    words = [data_rf.iloc[i, -1], data_lgb.iloc[i, -1], data_svm.iloc[i, -1]]
    collection_words = Counter(words)
    # print(collection_words)
    # print(len(collection_words))
    if len(collection_words) != 3:
        # print(i)
        # print(max(collection_words,key=collection_words.get))
        index1.append(i)
        index1_label.append(max(collection_words, key=collection_words.get))

        # ans['测试'] = max(collection_words,key=collection_words.get)
    # 这种情况是少数服从多数的投票机制

    elif len(collection_words) == 3:
        # 这种情况是少数服从多数的投票机制
        print(i)
        print(max(collection_words, key=collection_words.get))
        print(collection_words)
        # ans['测试'] = data_rf.iloc[i,-1]
        # 这种情况是，每个模型预测的都不一样，以最高准确率的模型为准
        index2.append(i)
        index2_label.append(data_rf.iloc[i, -1])
    # print(type(collection_words))

# In[17]:


ans['测试'] = np.zeros(1000)

# In[18]:


ans.iloc[index1, 1] = index1_label
ans.iloc[index2, 1] = index2_label

# In[19]:


ans['测试'].value_counts()

# In[20]:


ans.columns = ['是否发生故障', '具体故障类别']

# In[21]:


ans

# <h2>3.获得最终结果</h2>

# In[22]:


data = pd.concat([data, ans], axis=1)
data

# In[23]:


data1 = data['具体故障类别'].value_counts()
data1 = data1.to_frame()
print(type(data1))
if os.access("故障数据/data_type.xlsx", os.F_OK):
    print ("Given file path is exist.")
    os.remove("故障数据/data_type.xlsx")
    print("delete sussessful")
outputpath = '故障数据/data_type.xlsx'
data1.to_excel(outputpath)

data2 = data['机器质量等级'].value_counts()
print(data['机器质量等级'].value_counts())

data2 = data2.to_frame()
print(type(data2))
if os.access("故障数据/data_grades.xlsx", os.F_OK):
    print ("Given file path is exist.")
    os.remove("故障数据/data_grades.xlsx")
    print("delete sussessful")
outputpath = '故障数据/data_grades.xlsx'
data2.to_excel(outputpath)

# In[24]:


revise = data[(data['具体故障类别'] != 'Normal') & (data['是否发生故障'] == 0)].index  # 获得指定的行

data.iloc[revise, -2] = 1
data[(data['具体故障类别'] != 'Normal') & (data['是否发生故障'] == 0)]  # 获得指定的行

# In[25]:


revise2 = data[(data['具体故障类别'] == 'Normal') & (data['是否发生故障'] == 1)].index  # 获得指定的行

# 19个判断为准确，确为1

data.iloc[revise2, -2] = 0

data[(data['具体故障类别'] == 'Normal') & (data['是否发生故障'] == 1)]

# In[27]:


data.to_excel('forecast.xlsx', index=0)


# <h2>4.可视化结果对比</h2>

# In[28]:


def visul_size(dataset, name):
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
    labels = ['Normal', 'HDF', 'TWF', 'OSF', 'PWF', 'RNF']
    colors = ['green', 'yellow', 'cyan', 'red', 'blue', 'magenta']
    plt.title(str(name) + '故障数量分布图(柱图)', fontsize=20)
    rect = plt.bar(range(len(bad_type)), bad_type, color=colors)
    autolabel(rect)
    plt.xticks(range(len(bad_type)), labels)
    plt.xlabel('故障类型', fontsize=15)
    plt.ylabel('故障数量', fontsize=15)
    plt.show()


# In[29]:


visul_size(data, '投票后')


# In[30]:


data_lgb['具体故障类别'].value_counts()

# In[31]:


visul_size(data_lgb, 'lgb')

# In[32]:


visul_size(data_rf, 'rf')

# In[33]:


visul_size(data_svm, 'svm')

# In[ ]:




