# -*- coding = utf-8 -*-
# @Time : 2023/3/14 11:28
# @Author : 张贵
# @File : svm二分类（中级机器设备）.py
# @Software : PyCharm


import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from time import time
import datetime
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score  # 相关评估指标
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import cohen_kappa_score  # 卡帕系数
from sklearn.metrics import hamming_loss  # 海明距离
from lightgbm import plot_importance  # 特征重要性
from time import time
import datetime
# import sys

# 魔法命令，绘图会好看
# get_ipython().run_line_magic('matplotlib', 'inline')

# 让jupyter中的图画上的中文都显示出来

plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

# <h2>1.数据查看</h2>

data_low = pd.read_csv('第二问数据集/中级机器设备.csv')


data_low['是否发生故障'] = data_low['是否发生故障'].astype('int')


# <h2>2.划分训练集和测试集</h2>

split = int(len(data_low) * 0.8)

data_train = data_low.iloc[:split, :-1]
data_test = data_low.iloc[split:, :-1]
data_test.reset_index(drop=True, inplace=True)

data_label = data_low.iloc[:split, -1]
data_label1 = data_low.iloc[split:, -1]
data_label1.reset_index(drop=True, inplace=True)

x_train = data_train  # 特征
y_train = data_label  # 训练集标签

x_test = data_test
y_test = data_label1  # 测试集标签

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# <h2>3.标准化</h2>

sc = StandardScaler()
xtrain = sc.fit_transform(x_train)  # 训练集标准化  特征
xtest = sc.transform(x_test)  # 测试集标准化  特征
ytrain = y_train
ytest = y_test

# <h2>4.模型调参</h2>

def model_parameter(xtrain, ytrain):
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    clf_score = {}
    for kernel in kernel:
        time0 = time()
        clf = SVC(kernel=kernel,
                  degree=1,
                  gamma="auto",
                  ).fit(xtrain, ytrain)

        print("The accuracy under kernel %s is %f" % (kernel, clf.score(xtest, ytest)))

        clf_score[kernel] = clf.score(xtest, ytest)

    kernels = max(clf_score, key=clf_score.get)  # 返回最大的得分作为核
    print(kernels)

    #     print("-" * 100)
    #     """
    #     gamma调参，当gamma变大时决策边界变得更不规则，开始围绕单个实例绕弯
    #     """
    #     score = []
    #     gamma_range = np.logspace(-10,1,50)
    #     for i in gamma_range:
    #         clf = SVC(kernel = 'rbf',gamma = i).fit(xtrain,ytrain)
    #         score.append(clf.score(xtest,ytest))
    #     print('The best score is ',max(score)," , and it's gamma is ",gamma_range[score.index(max(score))])
    #     plt.plot(gamma_range,score) # 画出学习曲线
    #     plt.show()

    #     gammas = gamma_range[score.index(max(score))]  # 返回最佳gamma值

    """
    c调参，有c意味着是软间隔分类。C比较像正则化参数对于逻辑回归的影响.
    噪音比较大的话，就用一个小c
    """
    score = []
    c_range = np.linspace(0.01, 30, 50)
    for i in c_range:
        clf = SVC(kernel=kernels, C=i).fit(xtrain, ytrain)
        score.append(clf.score(xtest, ytest))
    print('The best score is ', max(score), " , and it's c is ", c_range[score.index(max(score))])

    cs = c_range[score.index(max(score))]

    return kernels, cs


kernels, cs = model_parameter(xtrain, ytrain)

# 训练
classifier = SVC(kernel=kernels, random_state=0, C=cs, decision_function_shape='ovr')
classifier.fit(xtrain, ytrain)

# Pridiction
y_pred = classifier.predict(xtest)


joblib.dump(classifier, '模型保存/save_model(问题二SVM中级).pkl')

# <h2>5.结果查看及相关评估指标</h2>

cnf = confusion_matrix(ytest, y_pred)


# Score
print(f'准确率：{classifier.score(xtest, ytest)}')

conf_matrix = pd.DataFrame(cnf, index=[x for x in range(0, 2)], columns=[x for x in range(0, 2)])

# plot size setting
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues", fmt='g')
plt.ylabel('真实标签', fontsize=18)
plt.xlabel('预测值', fontsize=18)
plt.title('中级机器二分类SVM测试集混淆矩阵可视化', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('confusion.pdf', bbox_inches='tight')
plt.show()

kappa = cohen_kappa_score(ytest, y_pred)  # (label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
print(f'卡帕系数为{kappa}')

ham_distance = hamming_loss(ytest, y_pred)
print(f'海明距离为{ham_distance}')

true = ytest
pre = y_pred

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

model = joblib.load('模型保存/save_model(问题二SVM中级).pkl')

model.predict(xtest)

# print(model.predict(xtest))




