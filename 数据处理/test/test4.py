# -*- coding = utf-8 -*-
# @Time : 2023/3/20 10:36
# @Author : 张贵
# @File : test4.py
# @Software : PyCharm

import pandas as pd

courses = {"语文": 80, "数学": 90, "英语": 85, "计算机": 100}

data = pd.Series(courses)

# Series转换为dataframe

df = pd.DataFrame(data, columns=["grade"])
print(df)
