# -*- coding = utf-8 -*-
# @Time : 2023/3/14 11:23
# @Author : 张贵
# @File : lg二分类（高级机器设备）.py
# @Software : PyCharm



import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler  # 归一化
from sklearn.preprocessing import OneHotEncoder  # 独热编码
from sklearn.model_selection import StratifiedKFold, KFold  # K折验证
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import LabelEncoder  # 标签专用，能够将分类转为分类数值
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score  # 相关评估指标
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import cohen_kappa_score  # 卡帕系数
from sklearn.metrics import hamming_loss  # 海明距离
from lightgbm import plot_importance  # 特征重要性
import seaborn as sns
import warnings
import joblib
import os

warnings.filterwarnings('ignore')
import multiprocessing

# 魔法命令，绘图会好看
# get_ipython().run_line_magic('matplotlib', 'inline')

# 让jupyter中的图画上的中文都显示出来

plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

# <h2>1.数据查看</h2>

# In[2]:


data_low = pd.read_csv('第二问数据集/高级机器设备.csv')

# In[3]:


data_low

# In[4]:


data_low['是否发生故障'] = data_low['是否发生故障'].astype('int')

# <h2>2.划分训练集和测试集</h2>

# In[5]:


split = int(len(data_low) * 0.8)

# In[6]:


data_train = data_low.iloc[:split, :-1]
data_test = data_low.iloc[split:, :-1]
data_test.reset_index(drop=True, inplace=True)

data_label = data_low.iloc[:split, -1]
data_label1 = data_low.iloc[split:, -1]
data_label1.reset_index(drop=True, inplace=True)

# In[7]:


x_train = data_train  # 特征
y_train = data_label  # 训练集标签

x_test = data_test
y_test = data_label1  # 测试集标签

# In[8]:


x_test.shape, y_test.shape

# In[9]:


x_train.shape, y_train.shape


# <h2>3.定义损失函数</h2>

# In[10]:


def abs_sum(y_pre, y_tru):
    y_pre = np.array(y_pre)
    y_tru = np.array(y_tru)
    loss = sum(sum(abs(y_pre - y_tru)))
    return loss


# <h2>4.建模</h2>

# In[11]:


def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2022
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    test = np.zeros((test_x.shape[0], 2))

    cv_scores = []
    onehot_encoder = OneHotEncoder(sparse=False)
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], \
        train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 2,
                'num_leaves': 2 ** 12 - 1,
                'feature_fraction': 1,
                'bagging_fraction': 1,
                'bagging_freq': 8,
                'learning_rate': 0.1,
                'seed': seed,
                'nthread': 28,
                'n_jobs': 24,
                'verbose': -1,
                'n_estimators': 3000,
                'max_depth': 12,
                'min_data_in_leaf': 30,  # 默认20，加大防止过拟合使用的,图像像左边移动
                'max_bin': 180,
                # 'reg_lambda':1,
                # 'reg_alpha' : 0.1

            }

            model = clf.train(params,
                              train_set=train_matrix,
                              valid_sets=valid_matrix,
                              num_boost_round=1000,
                              verbose_eval=100,
                              early_stopping_rounds=50)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)  # 验证集预测
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)  # 测试集预测
            # lgb_score = accuracy_score(data_test_label,test_pred.astype(np.int64))
            # lgb_auc = roc_auc_score(data_test_label,test_pred.astype(np.int64))
            # print(f'准确率为:{lgb_score}')
            # print(f'auc为:{lgb_auc}')
            joblib.dump(model, '模型保存/save_model(问题二lgb高级).pkl')

        val_y = np.array(val_y).reshape(-1, 1)
        val_y = onehot_encoder.fit_transform(val_y)
        # print(test_pred)
        test += test_pred
        score = abs_sum(val_y, val_pred)
        cv_scores.append(score)
        print('得分：')
        print(cv_scores)

    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))

    test = test / kf.n_splits  # 返回最终的预测结果

    return test


# In[12]:


def lgb_model(x_train, y_train, x_test):
    lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_test


# In[13]:


lgb_test = lgb_model(x_train, y_train, x_test)  # 输入训练集特征和标签， 测试集特征

# <h2>5.结果查看</h2>

# <h4>5.1测试集结果</h4>

# In[14]:


pre = np.argmax(lgb_test, axis=1)  # 测试结果
print(f'预测结果为{pre}')

# In[15]:


true = np.array(y_test)  # 真实结果
print(f'真实标签为{true}')

# In[16]:


# 计算混淆矩阵
cnf = confusion_matrix(true, pre, labels=[x for x in range(0, 2)])
cnf

# In[17]:


conf_matrix = pd.DataFrame(cnf, index=[x for x in range(0, 2)], columns=[x for x in range(0, 2)])

# plot size setting
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues", fmt='g')
plt.ylabel('真实标签', fontsize=18)
plt.xlabel('预测值', fontsize=18)
plt.title('高级机器设备二分类lgb测试集混淆矩阵可视化', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('confusion.pdf', bbox_inches='tight')
plt.show()

# <h2>6.各种评估指标</h2>

# In[18]:


kappa = cohen_kappa_score(true, pre)  # (label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
print(f'卡帕系数为{kappa}')

# In[19]:


ham_distance = hamming_loss(true, pre)
print(f'海明距离为{ham_distance}')

# In[20]:


print('------Weighted------')
print('Weighted precision', precision_score(true, pre, average='weighted'))
print('Weighted recall', recall_score(true, pre, average='weighted'))
print('Weighted f1-score', f1_score(true, pre, average='weighted'))
print('------Macro------')
print('Macro precision', precision_score(true, pre, average='macro'))
print('Macro recall', recall_score(true, pre, average='macro'))
print('Macro f1-score', f1_score(true, pre, average='macro'))
print('------Micro------')
print('Micro precision', precision_score(true, pre, average='micro'))
print('Micro recall', recall_score(true, pre, average='micro'))
print('Micro f1-score', f1_score(true, pre, average='micro'))


# <h4>6.2评估指标可视化</h4>

# In[21]:


# 编辑图例
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # 设置图例字体、位置、数值等等
        plt.text(rect.get_x(), 1.01 * height, '%s' %
                 float(height), size=15, family="Times new roman")


evaluation_indicators = [round(precision_score(true, pre, average='macro'), 3),
                         round(recall_score(true, pre, average='macro'), 3),
                         round(f1_score(true, pre, average='macro'), 3)]
plt.figure(figsize=(12, 8))
labels = ['Precision', 'Recall', 'F1_score']
plt.title('高级机器设备测试集评估指标可视化', fontsize=20)
colors = ['red', 'yellow', 'cyan']
rect = plt.bar(range(len(evaluation_indicators)), evaluation_indicators, color=colors)
autolabel(rect)
plt.xticks(range(len(evaluation_indicators)), labels)
plt.xlabel('评估指标', fontsize=15)
plt.ylabel('准确率', fontsize=15)

# <h2>7.调用保存好的模型</h2>

# <h4>7.1看一下训练集的效果</h4>

# In[22]:


model = joblib.load('模型保存/save_model(问题二lgb高级).pkl')

# In[23]:


model.predict(x_train)  # 返回的是每一类的概率   这里填入测试集的特征

# In[24]:


plot_importance(model, max_num_features=5, xlabel='特征重要性', ylabel='特征',
                title='高级机器设备特征对标签的重要程度可视化')

# In[25]:


train_outcome = np.argmax(model.predict(pd.DataFrame(x_train)), axis=1)  # 取最大 获得预测结果
train_outcome

# In[26]:


train_trueLabel = np.array(y_train)  # 真实标签
train_trueLabel

# In[27]:


# 测试集效果
from sklearn.metrics import confusion_matrix

cnf = confusion_matrix(train_trueLabel, train_outcome, labels=[x for x in range(0, 2)])
cnf

conf_matrix = pd.DataFrame(cnf, index=[x for x in range(0, 2)], columns=[x for x in range(0, 2)])

# plot size setting
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues", fmt='g')
plt.ylabel('真实值', fontsize=18)
plt.xlabel('预测值', fontsize=18)
plt.title('高级机器设备二分类lgb训练集混淆矩阵可视化', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('confusion.pdf', bbox_inches='tight')
plt.show()

# In[28]:


kappa = cohen_kappa_score(train_trueLabel, train_outcome)  # (label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
print(f'卡帕系数为{kappa}')

# In[29]:


ham_distance = hamming_loss(train_trueLabel, train_outcome)
print(f'海明距离为{ham_distance}')

# In[30]:


print('------Weighted------')
print('Weighted precision', precision_score(train_trueLabel, train_outcome, average='weighted'))
print('Weighted recall', recall_score(train_trueLabel, train_outcome, average='weighted'))
print('Weighted f1-score', f1_score(train_trueLabel, train_outcome, average='weighted'))
print('------Macro------')
print('Macro precision', precision_score(train_trueLabel, train_outcome, average='macro'))
print('Macro recall', recall_score(train_trueLabel, train_outcome, average='macro'))
print('Macro f1-score', f1_score(train_trueLabel, train_outcome, average='macro'))
print('------Micro------')
print('Micro precision', precision_score(train_trueLabel, train_outcome, average='micro'))
print('Micro recall', recall_score(train_trueLabel, train_outcome, average='micro'))
print('Micro f1-score', f1_score(train_trueLabel, train_outcome, average='micro'))

# <h4>7.2预测一下问题四的低级机器设备二分类</h4>

# In[31]:


aa = pd.read_excel('原数据/forecast.xlsx')
aa

# In[32]:


aa['厂房室温（K）'] = aa['厂房室温（K）'] - 273.15
aa['机器温度（K）'] = aa['机器温度（K）'] - 273.15

# In[33]:


aa.rename(columns={'厂房室温（K）': '厂房室温（℃）', '机器温度（K）': '机器温度（℃）'}, inplace=True)

# In[34]:


aa.drop(['机器编号', '统一规范代码', '使用时长（min）'], axis=1, inplace=True)

# In[35]:


aa = aa[aa['机器质量等级'] == 'H']
aa

# In[36]:


P = (aa['转速（rpm）'] * aa['扭矩（Nm）']) / 9550
aa.insert(5, '功率(KW)', P)

# In[37]:


aa.drop(['机器质量等级'], axis=1, inplace=True)

# In[38]:


aa

# In[39]:


testXX = np.argmax(model.predict(pd.DataFrame(aa)), axis=1)  # 取最大   #输入特征，返回预测结果

# In[40]:


testXX

# In[41]:


C = pd.DataFrame(testXX)  # 把预测结果送入pd.DataFrame中，统计0，1的数量
C

# In[42]:


C[0].value_counts()

# In[43]:


nums = []
for i in C[0].value_counts():
    # print(i)
    nums.append(i)

# In[44]:


plt.figure(figsize=(12, 7))
plt.style.use('fivethirtyeight')
plt.pie(nums, labels=['没有故障', '有故障'], autopct='%1.1f%%', counterclock=False, startangle=90)
plt.title('高级机器设备等级数量分布图(饼图)')
plt.show()

# In[ ]:




