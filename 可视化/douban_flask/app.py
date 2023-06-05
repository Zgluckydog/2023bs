from flask import Flask,render_template
import sqlite3
app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/')
def index():  # put application's code here
    return render_template("index.html")

@app.route('/index')
def home():
    # return render_template("index.html")
    return index()

@app.route('/movie')
def movie():
    datalist = []
    con = sqlite3.connect("movie.db")
    cur = con.cursor()
    sql = "select * from movieTop250"
    data = cur.execute(sql)
    for item in data:
        datalist.append(item)
    cur.close()
    con.close()
    return render_template("movie.html",movies = datalist)

@app.route('/score')
def score():
    num = []
    false_type = []
    con = sqlite3.connect('C:\\Users\\86173\\Desktop\\毕设\\数据处理\\故障数据\\save_data')
    cur = con.cursor()
    sql = "select false_type,num from data_low"
    data = cur.execute(sql)
    for item in data:
        num.append(item[1])
        false_type.append(item[0])
    cur.close()
    con.close()
    return render_template("score.html",false_type = false_type, num = num)

@app.route('/word')
def word():
    return render_template("word.html")
@app.route('/team')
def team():
    return render_template("team.html")

if __name__ == '__main__':
    app.run()
