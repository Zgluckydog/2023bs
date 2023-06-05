# -*- coding = utf-8 -*-
# @Time : 2023/3/29 15:24
# @Author : 张贵
# @File : test_db.py
# @Software : PyCharm

import sqlite3
num = []
false_type = []
conn = sqlite3.connect('C:\\Users\\86173\\Desktop\\毕设\\数据处理\\故障数据\\save_data')
cur = conn.cursor()
sql = "select false_type,num from data_low"
data = cur.execute(sql)
for item in data:
    num.append(item[1])
    false_type.append(item[0])
print(num)
print(false_type)