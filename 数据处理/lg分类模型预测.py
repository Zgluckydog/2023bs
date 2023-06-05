# -*- coding = utf-8 -*-
# @Time : 2023/3/14 23:14
# @Author : 张贵
# @File : lg分类模型预测.py
# @Software : PyCharm


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score  # 相关评估指标
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import cohen_kappa_score  # 卡帕系数
from sklearn.metrics import hamming_loss  # 海明距离
from lightgbm import plot_importance  # 特征重要性
import joblib

# 魔法命令，绘图会好看
# get_ipython().run_line_magic('matplotlib', 'inline')

# 让jupyter中的图画上的中文都显示出来

plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

# <h2>1.数据集划分</h2>

# In[2]:


data = pd.read_excel('原数据/forecast.xlsx')

# In[3]:


data.head()

# <h4>1.1低级机器设备数据集</h4>

# In[4]:


data_low = data.copy()

data_low.drop(['机器编号', '统一规范代码', '厂房室温（K）', '机器温度（K）'], axis=1, inplace=True)

data_low = data_low[data_low['机器质量等级'] == 'L']
P = (data_low['转速（rpm）'] * data_low['扭矩（Nm）']) / 9550
data_low.insert(3, '功率(KW)', P)
data_low.drop(['机器质量等级'], axis=1, inplace=True)

# In[5]:


data_low.to_csv('第四问数据集/低级机器设备.csv')

# In[6]:


data_low

# <h4>1.2中级机器设备数据集</h4>

# In[7]:


data_med = data.copy()
data_med['厂房室温（K）'] = data_med['厂房室温（K）'] - 273.15
data_med['机器温度（K）'] = data_med['机器温度（K）'] - 273.15
data_med.rename(columns={'厂房室温（K）': '厂房室温（℃）', '机器温度（K）': '机器温度（℃）'}, inplace=True)
data_med.drop(['机器编号', '统一规范代码'], axis=1, inplace=True)
data_med = data_med[data_med['机器质量等级'] == 'M']
P = (data_med['转速（rpm）'] * data_med['扭矩（Nm）']) / 9550
data_med.insert(5, '功率(KW)', P)
data_med.drop(['机器质量等级'], axis=1, inplace=True)

# In[8]:


data_med

# In[9]:


data_med.to_csv('第四问数据集/中级机器设备.csv')

# <h4>1.3高级机器设备数据集</h4>

# In[10]:


data_high = data.copy()
data_high['厂房室温（K）'] = data_high['厂房室温（K）'] - 273.15
data_high['机器温度（K）'] = data_high['机器温度（K）'] - 273.15
data_high.rename(columns={'厂房室温（K）': '厂房室温（℃）', '机器温度（K）': '机器温度（℃）'}, inplace=True)
data_high.drop(['机器编号', '统一规范代码', '使用时长（min）'], axis=1, inplace=True)
data_high = data_high[data_high['机器质量等级'] == 'H']
P = (data_high['转速（rpm）'] * data_high['扭矩（Nm）']) / 9550
data_high.insert(5, '功率(KW)', P)
data_high.drop(['机器质量等级'], axis=1, inplace=True)

# In[11]:


data_high

# In[12]:


data_high.to_csv('第四问数据集/高级机器设备.csv')


# <h2>2.二分类模型预测结果</h2>

# In[13]:


def model_load(model_path, pre_dataset, level):
    # 导入模型，要预测的数据集,机器等级
    model = joblib.load(model_path)
    plot_importance(model, max_num_features=6, xlabel='特征重要性', ylabel='特征',
                    title=str(level) + '机器设备特征对标签的重要程度可视化')
    prediction = np.argmax(model.predict(pd.DataFrame(pre_dataset)), axis=1)  # 预测结果
    print(f'预测结果如下：')
    print(prediction)

    return prediction


# <h4>2.1低级机器设备</h4>

# In[14]:


path = '模型保存/save_model(问题二lgb低级).pkl'
bindary_class_low = model_load(path, data_low, '低级')

# <h4>2.2中级机器设备</h4>

# In[15]:


path = '模型保存/save_model(问题二lgb中级).pkl'
bindary_class_med = model_load(path, data_med, '中级')

# <h4>2.3高级机器设备</h4>

# In[16]:


path = '模型保存/save_model(问题二lgb高级).pkl'
bindary_class_high = model_load(path, data_high, '高级')

# <h4>2.4结果放入原数据中</h4>

# In[17]:


data_low['索引'] = data_low.index
data_med['索引'] = data_med.index
data_high['索引'] = data_high.index

# In[18]:


data_low['是否发生故障'] = bindary_class_low
data_med['是否发生故障'] = bindary_class_med
data_high['是否发生故障'] = bindary_class_high

# In[19]:


data_low

# In[20]:


bindary_predict = pd.concat([data_low.iloc[:, -2:], data_med.iloc[:, -2:], data_high.iloc[:, -2:]], axis=0)
bindary_predict.sort_values('索引', inplace=True)

# In[21]:


data['是否发生故障'] = bindary_predict.iloc[:, 1]

# In[22]:


data

# <h2>3.多分类预测结果</h2>

# <h4>3.1低级机器设备</h4>

# In[23]:


data_low

# In[24]:


path = '模型保存/save_model(问题三lgb低级).pkl'
mutil_class_low = model_load(path, data_low.iloc[:, :4], '低级')

# <h4>3.2中级机器设备</h4>

# In[25]:


path = '模型保存/save_model(问题三lgb中级).pkl'
mutil_class_med = model_load(path, data_med.iloc[:, :6], '中级')

# <h4>3.3高级机器设备</h4>

# In[26]:


path = '模型保存/save_model(问题三lgb高级).pkl'
mutil_class_high = model_load(path, data_high.iloc[:, :5], '高级')

# <h4>3.4结果放入原数据中</h4>

# In[27]:


data_low['索引2'] = data_low.index
data_med['索引2'] = data_med.index
data_high['索引2'] = data_high.index

# In[28]:


data_low['具体故障类别'] = mutil_class_low
data_med['具体故障类别'] = mutil_class_med
data_high['具体故障类别'] = mutil_class_high

# In[29]:


data_low['具体故障类别'].value_counts()

# In[30]:


# 改为字符串
data_low.iloc[:, -1].replace([0, 1, 2, 3, 4, 5], ['Normal', 'OSF', 'HDF', 'PWF', 'TWF', 'RNF'], inplace=True)
data_med.iloc[:, -1].replace([0, 1, 2, 3, 4, 5], ['Normal', 'OSF', 'HDF', 'PWF', 'TWF', 'RNF'], inplace=True)
data_high.iloc[:, -1].replace([0, 1, 2, 3], ['Normal', 'HDF', 'PWF', 'TWF'], inplace=True)

# In[31]:


mutil_predict = pd.concat([data_low.iloc[:, -2:], data_med.iloc[:, -2:], data_high.iloc[:, -2:]], axis=0)
mutil_predict.sort_values('索引2', inplace=True)

# In[32]:


mutil_predict

# In[33]:


data['具体故障类别'] = mutil_predict.iloc[:, 1]

# <h2>4.获得总预测结果</h2>

# In[34]:


data

# In[35]:


# data.to_csv('第四题lgb预测结果.csv',index = 0)


# <h4>故障修改为一致性，以多分类结果为准</h4>

# In[36]:


data.columns

# In[37]:


revise = data[(data['具体故障类别'] != 'Normal') & (data['是否发生故障'] == 0)].index  # 获得指定的行
# 33个多分类判断为错误，确为0
data.iloc[revise, -2] = 1
data[(data['具体故障类别'] != 'Normal') & (data['是否发生故障'] == 0)]  # 获得指定的行

# In[38]:


revise2 = data[(data['具体故障类别'] == 'Normal') & (data['是否发生故障'] == 1)].index  # 获得指定的行

# 19个判断为准确，确为1

data.iloc[revise2, -2] = 0

data[(data['具体故障类别'] == 'Normal') & (data['是否发生故障'] == 1)]

# In[39]:
plt.show()

data.to_csv('第四题lgb修正故障后的预测结果.csv', index=0)

