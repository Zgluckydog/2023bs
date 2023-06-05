# -*- coding = utf-8 -*-
# @Time : 2023/4/12 13:28
# @Author : 张贵
# @File : save_data_test.py.py
# @Software : PyCharm
import sqlite3
import xlwt
import pandas as pd

def main():
    dbpath = "save_data"
    df = pd.read_excel(r"C:\Users\86173\Desktop\毕设\数据处理\故障数据\data_test.xlsx")
    print(df)
    saveDataDB(df,dbpath)

def init_db(dbpath):
    sql = '''
        create table IF NOT EXISTS data_test
        (   
        id integer primary key autoincrement,
        data_type varchar,
        predict_data numeric,
        ture_data numeric,
        outcome_data varchar
        )
    '''
    conn = sqlite3.connect(dbpath)
    cursor = conn.cursor()
    cursor.execute(sql)
    sql = 'delete from data_test'
    cursor.execute(sql)
    conn.commit()
    conn.close()

def saveDataDB(df,dbpath):
    init_db(dbpath)
    conn = sqlite3.connect(dbpath)
    cur = conn.cursor()
    for i in range(len(df)):
        id = df.iloc[i][0]
        data_type = df.iloc[i][1]
        predict_data = df.iloc[i][2]
        true_data = df.iloc[i][3]
        outcome_data = df.iloc[i][4]
        sql = f"INSERT INTO data_test SELECT {id},'{data_type}',{predict_data},{true_data},'{outcome_data}'"
        print(sql)

        cur.execute(sql)
        conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":  # 当程序执行时
    # 调用函数
    # init_db("datatest.db")

    main()