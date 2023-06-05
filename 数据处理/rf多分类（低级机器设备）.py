# -*- coding = utf-8 -*-
# @Time : 2023/3/14 15:21
# @Author : 张贵
# @File : rf多分类（低级机器设备）.py
# @Software : PyCharm


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# In[145]:


data = pd.read_csv(r"第三问数据集/低等级机器设备.csv", index_col=0)

data

# In[146]:


data['转速（rpm）'] = data.index

data.index = range(0, len(data))

data.info()

data

# data[(data['使用时长（min）']==0)]


# In[147]:


# 删除对预测的y没有关系的列
# data.drop(["统一规范代码","机器质量等级","具体故障类别"],inplace=True,axis=1)

# print(data['厂房室温（K）'].min(),data['厂房室温（K）'].max())
#
# data_norm = (data['厂房室温（K）']-data['厂房室温（K）'].min()) / (data['厂房室温（K）'].max()-data['厂房室温（K）'].min())
# data['厂房室温（K）']=data_norm
#
# data_norm = (data['机器温度（K）']-data['机器温度（K）'].min()) / (data['机器温度（K）'].max()-data['机器温度（K）'].min())
# data['机器温度（K）']=data_norm

# data_norm = (data['转速（rpm）']-data['转速（rpm）'].min()) / (data['转速（rpm）'].max()-data['转速（rpm）'].min())
# data['转速（rpm）']=data_norm
#
# data_norm = (data['扭矩（Nm）']-data['扭矩（Nm）'].min()) / (data['扭矩（Nm）'].max()-data['扭矩（Nm）'].min())
# data['扭矩（Nm）']=data_norm
#
# data_norm = (data['功率（KW）']-data['功率（KW）'].min()) / (data['功率（KW）'].max()-data['功率（KW）'].min())
# data['功率（KW）']=data_norm
#
# data_norm = (data['使用时长（min）']-data['使用时长（min）'].min()) / (data['使用时长（min）'].max()-data['使用时长（min）'].min())
# data['使用时长（min）']=data_norm

data.head()

# In[148]:


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

# In[149]:


X = data.iloc[:, data.columns != "具体故障类别"]
y = data.iloc[:, data.columns == "具体故障类别"]

# In[150]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 修正测试集和训练集的索引
for i in [x_train, x_test, y_train, y_test]:
    i.index = range(i.shape[0])

# 查看分好的训练集和测试集
x_train.head()

# In[151]:


import joblib
# 使用单-决策树进行模型训练以及预测分析。
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=5, random_state=25)
dtc.fit(x_train, y_train)
dtc_y_pred = dtc.predict(x_test)

# 保存模型
# joblib.dump(dtc,'model_3_low_dt.pkl')
# model = joblib.load('模型保存/save_model.pkl')

# 使用随机森林分类器进行集成模型的训练以及预测分析。
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)

# 保存模型
joblib.dump(rfc, '模型保存/save_model(问题三rf低级).pkl')

# 使用梯度提升决策树进行集成模型的训练以及预测分析。
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_y_pred = gbc.predict(x_test)

# In[152]:


# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(rfc, x_test, y_test, cmap="Blues")
plt.ylabel('真实标签', fontsize=10)
plt.xlabel('预测值', fontsize=10)
plt.title('低级机器多分类rf测试集混淆矩阵可视化', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# In[153]:


from sklearn.metrics import cohen_kappa_score, hamming_loss

kappa = cohen_kappa_score(y_test, rfc_y_pred)  # (label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
print(f'卡帕系数为{kappa}')
ham_distance = hamming_loss(y_test, rfc_y_pred)
print(f'海明距离为{ham_distance}')

# In[154]:


# 从sklearn .metrics导人classification report。
from sklearn.metrics import classification_report

# 输出单一决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of decision tree is', dtc.score(x_test, y_test))
print(classification_report(dtc_y_pred, y_test))
print(dtc.feature_importances_)

# In[155]:


# 输出随机森林分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of random forest classifier is', rfc.score(x_test, y_test))
print(classification_report(rfc_y_pred, y_test))

print(rfc.feature_importances_)

# In[156]:


# 输出梯度提升决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of gradient tree boosting is', gbc.score(x_test, y_test))
print(classification_report(gbc_y_pred, y_test))

# In[157]:


tr = []
te = []
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25
                                 , max_depth=i + 1
                                 # ,criterion="entropy"
                                 )
    clf = clf.fit(x_train, y_train)
    score_tr = clf.score(x_train, y_train)
    score_te = cross_val_score(clf, X, y, cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print(max(te))
plt.plot(range(1, 11), tr, color="red", label="train")
plt.plot(range(1, 11), te, color="blue", label="test")
plt.xticks(range(1, 11))
plt.legend()
plt.show()

# In[158]:


from sklearn import tree

tree.plot_tree(dtc)
plt.figure()
plt.show()

# In[159]:


from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
import graphviz
import pydotplus
import os

from matplotlib import pyplot as plt

# 魔法命令，绘图会好看
# get_ipython().run_line_magic('matplotlib', 'inline')

# 让jupyter中的图画上的中文都显示出来

plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

os.environ["Path"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
dot_data = export_graphviz(dtc
                           , feature_names=['转速（rpm）', '扭矩（Nm）', '功率（KW）', '使用时长（min）']
                           , class_names=["0", "1", "2", "3", "4", "5"]
                           , filled=True
                           , rounded=True
                           )
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# In[160]:


# 魔法命令，绘图会好看
# get_ipython().run_line_magic('matplotlib', 'inline')

# 让jupyter中的图画上的中文都显示出来

plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False
# from xgboost import XGBClassifier
#
# model = XGBClassifier()
# model.fit(X_train_scaled, y_train)
importances = pd.DataFrame(data={
    'Attribute': x_train.columns,
    'Importance': rfc.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)
# 可视化
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()

# In[161]:


names = ["扭矩（Nm）", "功率（Kw）", "使用时长（min）", "转速（rpm）"]

print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), names),
             reverse=True))

# In[162]:


from sklearn.linear_model import LogisticRegression

# x_train_norm = (data['转速（rpm）']-data['转速（rpm）'].min()) / (data['转速（rpm）'].max()-data['转速（rpm）'].min())
# x_train['转速（rpm）']=x_train_norm
#
# x_train_norm = (data['扭矩（Nm）']-data['扭矩（Nm）'].min()) / (data['扭矩（Nm）'].max()-data['扭矩（Nm）'].min())
# x_train['扭矩（Nm）']=x_train_norm
#
# x_train_norm = (data['功率（KW）']-data['功率（KW）'].min()) / (data['功率（KW）'].max()-data['功率（KW）'].min())
# x_train['功率（KW）']=x_train_norm
#
# x_train_norm = (data['使用时长（min）']-data['使用时长（min）'].min()) / (data['使用时长（min）'].max()-data['使用时长（min）'].min())
# x_train['使用时长（min）']=x_train_norm

model = LogisticRegression()
model.fit(x_train, y_train)
importances = pd.DataFrame(data={
    'Attribute': x_train.columns,
    'Importance': model.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)
# 可视化
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()

# In[163]:


from sklearn.decomposition import PCA

pca = PCA().fit(x_train)
# 可视化
plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
plt.title('Cumulative explained variance by number of principal components', size=20)
plt.show()

# In[164]:


import numpy as np

loadings = pd.DataFrame(
    data=pca.components_.T * np.sqrt(pca.explained_variance_),
    columns=[f'PC{i}' for i in range(1, len(x_train.columns) + 1)],
    index=x_train.columns
)
loadings.head()
pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
pc1_loadings = pc1_loadings.reset_index()
pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
plt.title('PCA loading scores (first principal component)', size=20)
plt.xticks(rotation='vertical')
plt.show()

