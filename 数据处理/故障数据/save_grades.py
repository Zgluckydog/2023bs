# -*- coding = utf-8 -*-
# @Time : 2023/4/2 19:31
# @Author : 张贵
# @File : save_grades.py.py
# @Software : PyCharm
import sqlite3
import xlwt
import pandas as pd

def main():
    dbpath = "save_data"
    df = pd.read_excel(r"C:\Users\86173\Desktop\毕设\数据处理\故障数据\num_grades.xlsx")
    print(df)
    saveDataDB(df,dbpath)

def init_db(dbpath):
    sql = '''
        create table IF NOT EXISTS num_grades
        (   
        num numeric
        )
    '''
    conn = sqlite3.connect(dbpath)
    cursor = conn.cursor()
    cursor.execute(sql)
    sql = 'delete from num_grades'
    cursor.execute(sql)
    conn.commit()
    conn.close()

def saveDataDB(df,dbpath):
    init_db(dbpath)
    conn = sqlite3.connect(dbpath)
    cur = conn.cursor()
    for i in range(len(df)):
        num = df.iloc[i][0]
        sql = f"INSERT INTO num_grades SELECT {num}"
        print(sql)

        cur.execute(sql)
        conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":  # 当程序执行时
    # 调用函数
    # init_db("datatest.db")

    main()