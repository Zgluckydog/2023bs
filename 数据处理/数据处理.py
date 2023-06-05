# -*- coding = utf-8 -*-
# @Time : 2023/3/13 13:00
# @Author : 张贵
# @File : 数据处理.py
# @Software : PyCharm
# !/usr/bin/env python
# coding: utf-8

import os
import xlwt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minepy import MINE
from sklearn.preprocessing import OrdinalEncoder  # 把特征转换成分类数值
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler  # 归一化
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
path ='C:\\Users\\86173\\Desktop'
# 魔法命令，绘图会好看
# get_ipython().run_line_magic('matplotlib', 'inline')

# 让jupyter中的图画上的中文都显示出来

plt.rcParams['font.sans-serif'] = ['Simhei']  # 改变默认字体
plt.rcParams['axes.unicode_minus'] = False

# 1.调用数据集

data = pd.read_excel('原数据/train data.xlsx')

data.head()
# print(data.head())

# 简单分析

data['机器编号'].value_counts()
data['统一规范代码'].value_counts()
data['是否发生故障'].value_counts()
data['具体故障类别'].value_counts()
data['机器质量等级'].value_counts()

#2.缺失值检测

data.isnull().sum()
# print(data.isnull().sum())

#3.异常值检测


data[(data['具体故障类别'] == 'Normal') & (data['是否发生故障'] == 1)]  # 获得指定的行


data[(data['使用时长（min）'] == 0) & (data['是否发生故障'] == 1)]



data.iloc[7506, 8] = 0
data.iloc[8015, 8] = 0
# data.iloc[4490,8] = 0
# data.iloc[5678,8] = 0
# data.iloc[8175,8] = 0


# 原始数据可视化探索



plt.figure(figsize=(12, 7))
plt.title('厂房和机器温度可视化', fontsize=20)
plt.plot(data['厂房室温（K）'], label='厂房室温（K）')
plt.plot(data['机器温度（K）'], label='机器温度（K）')
plt.xlabel('样本', fontsize=15)
plt.ylabel('温度(K)', fontsize=15)
plt.legend()
plt.show()



fig = plt.figure(figsize=(12, 7))
plt.title('机器转速可视化', fontsize=20)
plt.plot(data['转速（rpm）'], label='转速（rpm）')
plt.xlabel('样本', fontsize=15)
plt.ylabel('转速', fontsize=15)
plt.legend()
plt.show()




plt.figure(figsize=(12, 7))
plt.title('机器扭矩可视化', fontsize=20)
plt.plot(data['扭矩（Nm）'], label='扭矩（Nm）')
plt.xlabel('样本', fontsize=15)
plt.ylabel('扭矩', fontsize=15)
plt.legend()
plt.show()


plt.figure(figsize=(12, 7))
plt.title('机器使用时长可视化', fontsize=20)
plt.plot(data['使用时长（min）'], label='使用时长（min）')
plt.xlabel('样本', fontsize=15)
plt.ylabel('时常', fontsize=15)
plt.legend()
plt.show()


data.columns


# 异常值可视化，异常值的处理也要考虑实际情况
check = ['厂房室温（K）', '机器温度（K）', '转速（rpm）', '扭矩（Nm）', '使用时长（min）']
fig = plt.figure(figsize=(15, 12))
fig.suptitle("箱型图异常值可视化", fontsize=20)
season = ['厂房温度', '机器温度', '机器转速', '机器扭矩', '机器使用时长']
for i in range(1, 6):
    axi = fig.add_subplot(3, 2, i)
    axi.boxplot(data[check[i - 1]])
    axi.set_title(season[i - 1], fontsize=15)

# 异常值可视化，异常值的处理也要考虑实际情况
check = ['厂房室温（K）', '机器温度（K）', '使用时长（min）']
fig = plt.figure(figsize=(15, 12))
fig.suptitle("箱型图异常值可视化", fontsize=20)
season = ['厂房温度', '机器温度', '机器使用时长']
for i in range(1, 4):
    axi = fig.add_subplot(2, 2, i)
    axi.boxplot(data[check[i - 1]])
    axi.set_title(season[i - 1], fontsize=15)
fig.savefig(path+'输出图片12.svg',format='svg',dpi=150)#输出

#扭矩和转速呈反比，因此我们要做归一化后查看是否全部是成反比的


data_ = data.copy()

# 归一化
scaler = MinMaxScaler()  # 实例化
scaler = scaler.fit(data_.iloc[:, 5:7])
result = scaler.transform(data_.iloc[:, 5:7])
data_.iloc[:, 5:7] = result


data_

data_.sort_values(['扭矩（Nm）'], ascending=True, inplace=True)  # 从小到大排序
data_.reset_index(drop=True, inplace=True)


fig = plt.figure(figsize=(12, 7))
plt.title('扭矩和转速的关系图', fontsize=20)
plt.xlabel('扭矩（Nm）', fontsize=15)
plt.ylabel('转速（rpm）', fontsize=15)
plt.plot(data_['扭矩（Nm）'], data_['转速（rpm）'])
plt.show()
fig.savefig(path+'输出图片12.svg',format='svg',dpi=150)#输出

#4.数据探索性分析

print(data['机器质量等级'])
quality_robot = []
for i in data['机器质量等级'].value_counts():
    quality_robot.append(i)
print(quality_robot)
qu = pd.DataFrame(quality_robot)
if os.access("故障数据/num_grades.xlsx", os.F_OK):
    print ("Given file path is exist.")
    os.remove("故障数据/num_grades.xlsx")
    print("delete sussessful")
outputpath = '故障数据/num_grades.xlsx'
qu.to_excel(outputpath)
# 编辑图例
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # 设置图例字体、位置、数值等等
        plt.text(rect.get_x(), 1.01 * height, '%s' %
                 float(height), size=15, family="Times new roman")


fig = plt.figure(figsize=(12, 7))
labels = ['低级质量', '中级质量', '高级质量']
colors = ['red', 'yellow', 'cyan']
plt.title('各机器设备等级数量分布图(柱图)', fontsize=20)
rect = plt.bar(range(len(quality_robot)), quality_robot, color=colors)
autolabel(rect)
plt.xticks(range(len(quality_robot)), labels)
plt.xlabel('质量等级', fontsize=15)
plt.ylabel('设备数量', fontsize=15)
plt.show()
fig.savefig(path+'输出图片1.svg',format='svg',dpi=150)#输出

fig = plt.figure(figsize=(12, 7))
plt.style.use('fivethirtyeight')
plt.pie(quality_robot, labels=labels, autopct='%1.1f%%', counterclock=False, startangle=90)
plt.title('各机器设备等级数量分布图(饼图)')
plt.show()
fig.savefig(path+'输出图片2.svg',format='svg',dpi=150)#输出
bad_type = []
for i in data['具体故障类别'].value_counts():
    bad_type.append(i)
ba = pd.DataFrame(bad_type)
if os.access("故障数据/bad_type.xlsx", os.F_OK):
    print ("Given file path is exist.")
    os.remove("故障数据/bad_type.xlsx")
    print("delete sussessful")
outputpath = '故障数据/bad_type.xlsx'
ba.to_excel(outputpath)
# 编辑图例
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # 设置图例字体、位置、数值等等
        plt.text(rect.get_x(), 1.01 * height, '%s' %
                 float(height), size=15, family="Times new roman")


fig = plt.figure(figsize=(12, 7))
labels = ['Normal', 'HDF', 'OSF', 'PWF', 'TWF', 'RNF']
colors = ['green', 'yellow', 'cyan', 'red', 'blue', 'magenta']
plt.title('故障数量分布图(柱图)', fontsize=20)
rect = plt.bar(range(len(bad_type)), bad_type, color=colors)
autolabel(rect)
plt.xticks(range(len(bad_type)), labels)
plt.xlabel('故障类型', fontsize=15)
plt.ylabel('故障数量', fontsize=15)
plt.show()
fig.savefig(path+'输出图片3.svg',format='svg',dpi=150)#输出


fig = plt.figure(figsize=(10, 10))
plt.style.use('fivethirtyeight')
plt.pie(bad_type, counterclock=False, startangle=90, explode=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
plt.title('故障数量分布图(饼图)')
plt.legend(labels, loc='best')
plt.show()
fig.savefig(path+'输出图片4.svg',format='svg',dpi=150)#输出
#5.机器质量和故障之间的关系



lowQua_good = data[(data['机器质量等级'] == 'L') & (data['是否发生故障'] == 0)]
lowQua_bad = data[(data['机器质量等级'] == 'L') & (data['是否发生故障'] == 1)]
medQua_good = data[(data['机器质量等级'] == 'M') & (data['是否发生故障'] == 0)]
medQua_bad = data[(data['机器质量等级'] == 'M') & (data['是否发生故障'] == 1)]
highQua_good = data[(data['机器质量等级'] == 'H') & (data['是否发生故障'] == 0)]
highQua_bad = data[(data['机器质量等级'] == 'H') & (data['是否发生故障'] == 1)]

def Qua_gorb(data1, data2, title ,name):
    # 可视化正品和次品饼图
    fig = plt.figure(figsize=(12, 7))
    plt.style.use('fivethirtyeight')
    plt.pie([len(data1), len(data2)], labels=['故障', '无故障'], autopct='%1.1f%%', counterclock=False, startangle=90)
    plt.title(str(title) + '等级机器设备故障分布图(饼图)')
    plt.show()
    fig.savefig(path + name, format='svg', dpi=150)  # 输出


Qua_gorb(lowQua_bad, lowQua_good, '低','输出图片5.svg')

Qua_gorb(medQua_bad, medQua_good, '中','输出图片6.svg')

Qua_gorb(highQua_bad, highQua_good, '高','输出图片7.svg')

#6.按设备等级划分数据集

data[data['机器质量等级'] == 'L']

P = (data['转速（rpm）'] * data['扭矩（Nm）']) / 9550
data.insert(7, '功率（KW）', P)

data['厂房室温（K）'] = data['厂房室温（K）'] - 273.15
data['机器温度（K）'] = data['机器温度（K）'] - 273.15

data.rename(columns={'厂房室温（K）': '厂房室温（℃）', '机器温度（K）': '机器温度（℃）'}, inplace=True)

fig = plt.figure(figsize=(12, 7))
plt.title('厂房和机器温度可视化', fontsize=20)
plt.plot(data['厂房室温（℃）'], label='厂房室温（℃）')
plt.plot(data['机器温度（℃）'], label='机器温度（℃）')
plt.xlabel('样本', fontsize=15)
plt.ylabel('温度（℃）', fontsize=15)
plt.legend()
plt.show()
fig.savefig(path + '输出图片19.svg', format='svg', dpi=150)  # 输出

data_low = data[data['机器质量等级'] == 'L']
data_med = data[data['机器质量等级'] == 'M']
data_high = data[data['机器质量等级'] == 'H']

data_low.reset_index(drop=True, inplace=True)
data_med.reset_index(drop=True, inplace=True)
data_high.reset_index(drop=True, inplace=True)

# data_low.to_csv('低级机器质量数据集.csv',index=0)
# data_med.to_csv('中级机器质量数据集.csv',index=0)
# data_high.to_csv('高级机器质量数据集.csv',index=0)


#7.逐一分析数据集---------特征间的关系

#功率和故障的关系

def p_qua(dataset, title,name):
    # 功率和质量之间的关系
    datas = dataset.copy()
    datas.sort_values(['功率（KW）'], ascending=True, inplace=True)  # 从小到大排序
    datas.reset_index(drop=True, inplace=True)

    fig = plt.figure(figsize=(12, 7))
    plt.title(str(title) + '功率故障关系图', fontsize=20)
    plt.scatter(range(len(datas['是否发生故障'])), datas['是否发生故障'])
    plt.ylabel('是否故障')
    plt.xlabel('功率从低到高排序')
    plt.show()
    fig.savefig(path + name, format='svg', dpi=150)  # 输出

p_qua(data_low, '低级机器设备','输出图片9.svg')

p_qua(data_med, '中级机器设备','输出图片10.svg')

p_qua(data_high, '高级机器设备','输出图片11.svg')


#温度和故障

def temp_qua(dataset, title ,name):
    # 温度和质量之间的关系
    datas = dataset.copy()
    datas.sort_values(['厂房室温（℃）'], ascending=True, inplace=True)  # 从小到大排序
    datas.reset_index(drop=True, inplace=True)

    fig=plt.figure(figsize=(12, 7))
    plt.title(str(title) + '机器温度故障关系图', fontsize=20)
    plt.scatter(range(len(datas['是否发生故障'])), datas['是否发生故障'])
    plt.ylabel('是否故障')
    plt.xlabel('机器温度从低到高排序')
    plt.show()
    fig.savefig(path + name, format='svg', dpi=150)  # 输出

temp_qua(data_low, '低级机器设备','输出图片13.svg')

temp_qua(data_med, '中级机器设备','输出图片14.svg')

temp_qua(data_high, '高级机器设备','输出图片15.svg')

#使用时长和故障

def time_qua(dataset, title,name):
    # 温度和质量之间的关系
    datas = dataset.copy()
    datas.sort_values(['使用时长（min）'], ascending=True, inplace=True)  # 从小到大排序
    datas.reset_index(drop=True, inplace=True)

    fig = plt.figure(figsize=(12, 7))
    plt.title(str(title) + '使用时长故障关系图', fontsize=20)
    plt.scatter(range(len(datas['是否发生故障'])), datas['是否发生故障'])
    plt.ylabel('是否故障')
    plt.xlabel('使用时长低到高排序')
    plt.show()
    fig.savefig(path + name, format='svg', dpi=150)  # 输出

time_qua(data_low, '低级机器设备','输出图片16.svg')

time_qua(data_med, '中级机器设备','输出图片17.svg')

time_qua(data_high, '高级机器设备','输出图片18.svg')

#8.MIC相关性查看每个数据集-----不能使用，因为是离散标签

def MIC_matirx(dataframe, mine):
    # 将MIC结果存入pd.Dataframe中
    data = np.array(dataframe)
    n = len(data[0, :])
    result = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            mine.compute_score(data[:, i], data[:, j])
            result[i, j] = mine.mic()
            result[j, i] = mine.mic()
    RT = pd.DataFrame(result, columns=dataframe.columns, index=dataframe.columns)
    return RT

# 相关度高的不一定需要合并。可以选择去除一个，或者是融合
def ShowHeatMap(DataFrame, title,name):
    # 可视化MIC结果
    colormap = plt.cm.RdBu
    ylabels = DataFrame.columns.values.tolist()
    f, ax = plt.subplots(figsize=(15, 12))
    ax.set_title(title, fontsize=25)
    sns.heatmap(DataFrame.astype(float),
                cmap=colormap,
                ax=ax,
                annot=True,
                yticklabels=ylabels,
                xticklabels=ylabels)
    plt.show()
    f.savefig(path + name, format='svg', dpi=150)  # 输出

mine = MINE(alpha=0.6, c=15)
mic_result = MIC_matirx(data_low.iloc[:, 3:10], mine)

ShowHeatMap(mic_result, '各特征之间的相关性矩阵(以低质量的为例)','输出图片8.svg')

mine = MINE(alpha=0.6, c=15)
mic_result = MIC_matirx(data_med.iloc[:, 3:10], mine)

ShowHeatMap(mic_result, '各特征之间的相关性矩阵(以中质量的为例)')


mine = MINE(alpha=0.6, c=15)
mic_result = MIC_matirx(data_high.iloc[:, 3:10], mine)

ShowHeatMap(mic_result, '各特征之间的相关性矩阵(以高质量的为例)')

#9.生成预测用的数据集

data_low.drop(['机器编号', '统一规范代码', '厂房室温（℃）', '机器温度（℃）'], axis=1, inplace=True)

data_med.drop(['机器编号', '统一规范代码', ], axis=1, inplace=True)

data_high.drop(['机器编号', '统一规范代码', '使用时长（min）'], axis=1, inplace=True)

data_low.to_csv('第二、三问数据集/低级机器设备数据集.csv', index=0)
data_med.to_csv('第二、三问数据集/中级机器设备数据集.csv', index=0)
data_high.to_csv('第二、三问数据集/高级机器设备数据集.csv', index=0)




