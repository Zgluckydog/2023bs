# -*- coding = utf-8 -*-
# @Time : 2023/3/14 23:25
# @Author : 张贵
# @File : rf分类模型预测.py
# @Software : PyCharm

# !/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score  # 相关评估指标
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import cohen_kappa_score  # 卡帕系数
from sklearn.metrics import hamming_loss  # 海明距离

import joblib

# 魔法命令，绘图会好看
# get_ipython().run_line_magic('matplotlib', 'inline')

# 让jupyter中的图画上的中文都显示出来

plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

# <h2>1.数据集划分</h2>

# In[88]:


data = pd.read_excel('原数据/forecast.xlsx')

# In[89]:


data.head()

# <h4>1.1低级机器设备数据集</h4>

# In[90]:


data_low = data.copy()

data_low.drop(['机器编号', '统一规范代码', '厂房室温（K）', '机器温度（K）'], axis=1, inplace=True)

data_low = data_low[data_low['机器质量等级'] == 'L']
P = (data_low['转速（rpm）'] * data_low['扭矩（Nm）']) / 9550
data_low.insert(3, '功率(KW)', P)
data_low.drop(['机器质量等级'], axis=1, inplace=True)

rpm = data_low.pop('转速（rpm）')
data_low.insert(loc=data_low.shape[1], column='转速（rpm）', value=rpm, allow_duplicates=False)
#
# aa.reindex(aa.columns[aa.columns != '转速（rpm）'].union(['转速（rpm）']), axis=1)
data_low.head()

# In[91]:


data_low.to_csv('第四问数据集/低级机器设备.csv')

# In[92]:


data_low

# <h4>1.2中级机器设备数据集</h4>

# In[93]:


data_med = data.copy()
data_med['厂房室温（K）'] = data_med['厂房室温（K）'] - 273.15
data_med['机器温度（K）'] = data_med['机器温度（K）'] - 273.15
data_med.rename(columns={'厂房室温（K）': '厂房室温（℃）', '机器温度（K）': '机器温度（℃）'}, inplace=True)
data_med.drop(['机器编号', '统一规范代码'], axis=1, inplace=True)
data_med = data_med[data_med['机器质量等级'] == 'M']
P = (data_med['转速（rpm）'] * data_med['扭矩（Nm）']) / 9550
data_med.insert(5, '功率(KW)', P)
data_med.drop(['机器质量等级'], axis=1, inplace=True)

# In[94]:


data_med

# In[95]:


data_med.to_csv('第四问数据集/中级机器设备.csv')

# <h4>1.3高级机器设备数据集</h4>

# In[96]:


data_high = data.copy()
data_high['厂房室温（K）'] = data_high['厂房室温（K）'] - 273.15
data_high['机器温度（K）'] = data_high['机器温度（K）'] - 273.15
data_high.rename(columns={'厂房室温（K）': '厂房室温（℃）', '机器温度（K）': '机器温度（℃）'}, inplace=True)
data_high.drop(['机器编号', '统一规范代码', '使用时长（min）'], axis=1, inplace=True)
data_high = data_high[data_high['机器质量等级'] == 'H']
P = (data_high['转速（rpm）'] * data_high['扭矩（Nm）']) / 9550
data_high.insert(5, '功率(KW)', P)
data_high.drop(['机器质量等级'], axis=1, inplace=True)

# In[97]:


data_high

# In[98]:


data_high.to_csv('第四问数据集/高级机器设备.csv')


# <h2>2.二分类模型预测结果</h2>

# In[99]:


def model_load(model_path, pre_dataset, level):
    # 导入模型，要预测的数据集,机器等级
    model = joblib.load(model_path)
    # plot_importance(model,max_num_features=6,xlabel='特征重要性',ylabel='特征',title=str(level) + '机器设备特征对标签的重要程度可视化')
    # prediction = np.argmax(model.predict(pd.DataFrame(pre_dataset)))#预测结果
    prediction = model.predict(pd.DataFrame(pre_dataset))
    print(f'预测结果如下：')
    print(prediction)

    return prediction


# <h4>2.1低级机器设备</h4>

# In[100]:


path = '模型保存/save_model(问题二RF低级).pkl'
bindary_class_low = model_load(path, data_low, '低级')

# <h4>2.2中级机器设备</h4>

# In[101]:


# path = '任务二/model_2_mid_rf.pkl'
# bindary_class_med =model_load(path,data_med,'中级')

path = '模型保存/save_model(问题二RF中级).pkl'
bindary_class_med = model_load(path, data_med, '中级')

# <h4>2.3高级机器设备</h4>

# In[102]:


# path = '任务二/model_2_high_rf.pkl'
# bindary_class_high =model_load(path,data_high,'高级')

path = '模型保存/save_model(问题二RF高级).pkl'
bindary_class_high = model_load(path, data_high, '高级')

# <h4>2.4结果放入原数据中</h4>

# In[103]:


data_low['索引'] = data_low.index
data_med['索引'] = data_med.index
data_high['索引'] = data_high.index

# In[104]:


data_low['是否发生故障'] = bindary_class_low
data_med['是否发生故障'] = bindary_class_med
data_high['是否发生故障'] = bindary_class_high

# In[105]:


data_low

# In[106]:


bindary_predict = pd.concat([data_low.iloc[:, -2:], data_med.iloc[:, -2:], data_high.iloc[:, -2:]], axis=0)
bindary_predict.sort_values('索引', inplace=True)

# In[107]:


data['是否发生故障'] = bindary_predict.iloc[:, 1]

# In[108]:


data

# <h2>3.多分类预测结果</h2>

# <h4>3.1低级机器设备</h4>

# In[109]:


path = '模型保存/save_model(问题三RF低级).pkl'

data_low.drop(['是否发生故障', '索引'], axis=1, inplace=True)

mutil_class_low = model_load(path, data_low, '低级')

# <h4>3.2中级机器设备</h4>

# In[110]:


path = '模型保存/save_model(问题三RF中级).pkl'

data_med.drop(['是否发生故障', '索引'], axis=1, inplace=True)

mutil_class_med = model_load(path, data_med, '中级')

# <h4>3.3高级机器设备</h4>

# In[111]:


path = '模型保存/save_model(问题三RF高级).pkl'

data_high.drop(['是否发生故障', '索引'], axis=1, inplace=True)

mutil_class_high = model_load(path, data_high, '高级')

# <h4>3.4结果放入原数据中</h4>

# In[112]:


data_low['索引2'] = data_low.index
data_med['索引2'] = data_med.index
data_high['索引2'] = data_high.index

# In[113]:


data_low['具体故障类别'] = mutil_class_low
data_med['具体故障类别'] = mutil_class_med
data_high['具体故障类别'] = mutil_class_high

# In[114]:


data_low['具体故障类别'].value_counts()

# In[115]:


# 改为字符串
data_low.iloc[:, -1].replace([0, 1, 2, 3, 4, 5], ['Normal', 'OSF', 'HDF', 'PWF', 'TWF', 'RNF'], inplace=True)
data_med.iloc[:, -1].replace([0, 1, 2, 3, 4, 5], ['Normal', 'OSF', 'HDF', 'PWF', 'TWF', 'RNF'], inplace=True)
data_high.iloc[:, -1].replace([0, 1, 2, 3], ['Normal', 'HDF', 'PWF', 'TWF'], inplace=True)

# In[116]:


mutil_predict = pd.concat([data_low.iloc[:, -2:], data_med.iloc[:, -2:], data_high.iloc[:, -2:]], axis=0)
mutil_predict.sort_values('索引2', inplace=True)

# In[117]:


mutil_predict

# In[118]:


data['具体故障类别'] = mutil_predict.iloc[:, 1]

# <h2>4.获得总预测结果</h2>

# In[119]:


data

# In[120]:


# data.to_csv('第四题预测结果.csv',index = 0)


# <h4>故障修改为一致性，以多分类结果为准</h4>

# In[121]:


data.columns

# In[122]:


revise = data[(data['具体故障类别'] != 'Normal') & (data['是否发生故障'] == 0)].index  # 获得指定的行
# 33个多分类判断为错误，确为0
data[(data['具体故障类别'] != 'Normal') & (data['是否发生故障'] == 0)]

# In[123]:


data.iloc[revise, -2] = 1
data[(data['具体故障类别'] != 'Normal') & (data['是否发生故障'] == 0)]  # 获得指定的行

# In[124]:


revise2 = data[(data['具体故障类别'] == 'Normal') & (data['是否发生故障'] == 1)].index  # 获得指定的行

data[(data['具体故障类别'] == 'Normal') & (data['是否发生故障'] == 1)]
# 19个判断为准确，确为1


# In[125]:


data.iloc[revise2, -2] = 0

data[(data['具体故障类别'] == 'Normal') & (data['是否发生故障'] == 1)]

# In[126]:
plt.show()

data.to_csv('第四题RF修正故障后的预测结果.csv', index=0)

# In[126]:




