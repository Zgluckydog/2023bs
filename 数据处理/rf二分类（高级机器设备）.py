# -*- coding = utf-8 -*-
# @Time : 2023/3/14 15:04
# @Author : 张贵
# @File : rf二分类（高级机器设备）.py
# @Software : PyCharm

# !/usr/bin/env python
# coding: utf-8

# # 预测

# In[79]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# In[80]:


data = pd.read_csv(r"第二问数据集/高级机器设备.csv")

data

# In[81]:


# data['转速（rpm）'] = data.index
#
# data.index = range(0,len(data))

data.info()

data

# data[(data['使用时长（min）']==0)]


# In[82]:


# 删除对预测的y没有关系的列
# data.drop(["统一规范代码","机器质量等级","具体故障类别"],inplace=True,axis=1)

# print(data['厂房室温（K）'].min(),data['厂房室温（K）'].max())
#
# data_norm = (data['厂房室温（℃）']-data['厂房室温（℃）'].min()) / (data['厂房室温（℃）'].max()-data['厂房室温（℃）'].min())
# data['厂房室温（℃）']=data_norm
#
# data_norm = (data['机器温度（℃）']-data['机器温度（℃）'].min()) / (data['机器温度（℃）'].max()-data['机器温度（℃）'].min())
# data['机器温度（℃）']=data_norm
#
# data_norm = (data['转速（rpm）']-data['转速（rpm）'].min()) / (data['转速（rpm）'].max()-data['转速（rpm）'].min())
# data['转速（rpm）']=data_norm
#
# data_norm = (data['扭矩（Nm）']-data['扭矩（Nm）'].min()) / (data['扭矩（Nm）'].max()-data['扭矩（Nm）'].min())
# data['扭矩（Nm）']=data_norm
#
# data_norm = (data['功率（KW）']-data['功率（KW）'].min()) / (data['功率（KW）'].max()-data['功率（KW）'].min())
# data['功率（KW）']=data_norm

# data_norm = (data['使用时长（min）']-data['使用时长（min）'].min()) / (data['使用时长（min）'].max()-data['使用时长（min）'].min())
# data['使用时长（min）']=data_norm

data.head()

# In[83]:


# 处理缺失值，对缺失值较多的列进行填补，有一些特征只确实一两个值，可以采取直接删除记录的方法
# data["Age"] = data["Age"].fillna(data["Age"].mean())            #用均值填补缺失值
# data = data.dropna()                                            #删掉有缺失值的行

# 将分类变量转换为数值型变量

# 将二分类变量转换为数值型变量
# astype能够将一个pandas对象转换为某种类型，和apply(int(x))不同，astype可以将文本类转换为数字，用这个方式可以很便捷地将二分类特征转换为0~1
# data["Sex"] = (data["Sex"]== "male").astype("int")

# 将三分类变量转换为数值型变量
# labels = data["机器质量等级"].unique().tolist()
# data["机器质量等级"] = data["机器质量等级"].apply(lambda x: labels.index(x))

# 查看处理后的数据集
data.head()

# In[84]:


X = data.iloc[:, data.columns != "是否发生故障"]
y = data.iloc[:, data.columns == "是否发生故障"]

# In[85]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 修正测试集和训练集的索引
for i in [x_train, x_test, y_train, y_test]:
    i.index = range(i.shape[0])

# 查看分好的训练集和测试集
x_train.head()

# In[86]:


import joblib
# 使用单-决策树进行模型训练以及预测分析。
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_y_pred = dtc.predict(x_test)

# 保存模型
# joblib.dump(dtc,'model_2_high_dt.pkl')

# 使用随机森林分类器进行集成模型的训练以及预测分析。
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)

# 保存模型
joblib.dump(rfc, '模型保存/save_model(问题二RF高级).pkl')

# 使用梯度提升决策树进行集成模型的训练以及预测分析。
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_y_pred = gbc.predict(x_test)

# In[87]:


from sklearn.metrics import cohen_kappa_score, hamming_loss

kappa = cohen_kappa_score(y_test, rfc_y_pred)  # (label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
print(f'卡帕系数为{kappa}')
ham_distance = hamming_loss(y_test, rfc_y_pred)
print(f'海明距离为{ham_distance}')

# In[88]:


# 从sklearn .metrics导人classification report。
from sklearn.metrics import classification_report

# 输出单一决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of decision tree is', dtc.score(x_test, y_test))
print(classification_report(dtc_y_pred, y_test))

# In[89]:


# 输出随机森林分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of random forest classifier is', rfc.score(x_test, y_test))
print(classification_report(rfc_y_pred, y_test))

# In[90]:


# 输出梯度提升决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of gradient tree boosting is', gbc.score(x_test, y_test))
print(classification_report(gbc_y_pred, y_test))

# In[91]:


# tr = []
# te = []
# for i in range(8):
#     clf = DecisionTreeClassifier(random_state=25
#                                  ,max_depth=i+1
#                                  # ,criterion="entropy"
#                                 )
#     clf = clf.fit(Xtrain, Ytrain)
#     score_tr = clf.score(Xtrain,Ytrain)
#     score_te = cross_val_score(clf,X,y,cv=10).mean()
#     tr.append(score_tr)
#     te.append(score_te)
# print(max(te))
# plt.plot(range(1,9),tr,color="red",label="train")
# plt.plot(range(1,9),te,color="blue",label="test")
# plt.xticks(range(1,9))
# plt.legend()
# plt.show()


# In[92]:


# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(rfc, x_test, y_test, cmap="Blues")
plt.ylabel('真实标签', fontsize=10)
plt.xlabel('预测值', fontsize=10)
plt.title('高级机器二分类rf测试集混淆矩阵可视化', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# In[92]:




