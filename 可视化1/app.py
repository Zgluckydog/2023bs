from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import  json
import sqlite3

app = Flask(__name__)

model = joblib.load("C:\\Users\\86173\\Desktop\\毕设\\数据处理\\模型保存\\save_model(问题二RF高级).pkl")
model1 = joblib.load("C:\\Users\\86173\\Desktop\\毕设\\数据处理\\模型保存\\save_model(问题二RF中级).pkl")
model2 = joblib.load("C:\\Users\\86173\\Desktop\\毕设\\数据处理\\模型保存\\save_model(问题二RF低级).pkl")
model3 = joblib.load("C:\\Users\\86173\\Desktop\\毕设\\数据处理\\模型保存\\save_model(问题三rf低级).pkl")
model4 = joblib.load("C:\\Users\\86173\\Desktop\\毕设\\数据处理\\模型保存\\save_model(问题三rf中级).pkl")
model5 = joblib.load("C:\\Users\\86173\\Desktop\\毕设\\数据处理\\模型保存\\save_model(问题三rf高级).pkl")
@app.route('/index')
def index():  # put application's code here
    num=[]
    false_type = []
    num1 = []
    num2 = []
    false_type1 = []
    num3 = []
    false_type2 = []
    num4 = []
    false_type3 = []
    num5 = []
    con = sqlite3.connect('C:\\Users\\86173\\Desktop\\毕设\\数据处理\\故障数据\\save_data')
    cur = con.cursor()
    sql = "select num from num_grades"
    data = cur.execute(sql)
    for item in data:
        num.append(item[0])

    sql = "select false_type,num from data_type"
    data1 = cur.execute(sql)
    for item in data1:
        num1.append(item[1])
        false_type.append(item[0])
    sql = "select num from bad_type"
    data2 = cur.execute(sql)
    for item in data2:
        num2.append(item[0])
    sql = "select * from data_low"
    data3 = cur.execute(sql)
    for item in data3:
        false_type1.append(item[0])
        num3.append(item[1])
    sql = "select * from data_med"
    data4 = cur.execute(sql)
    for item in data4:
        false_type2.append(item[0])
        num4.append(item[1])
    sql = "select * from data_high"
    data5 = cur.execute(sql)
    for item in data5:
        false_type3.append(item[0])
        num5.append(item[1])
    cur.close()
    con.close()
    return render_template("index.html" ,num = num,num1 = num1,false_type = false_type,num2 = num2,false_type1 = false_type1,num3 = num3,false_type2 = false_type2,num4 = num4,false_type3 = false_type3,num5 = num5  )

@app.route('/flot')
def flot():
    data3 = []
    data4 = []
    con2 = sqlite3.connect('C:\\Users\\86173\\Desktop\\毕设\\数据处理\\故障数据\\save_data')
    cur2 = con2.cursor()
    sql2 = "select predict_data,ture_data from data_test"
    data5 = cur2.execute(sql2)
    for item in data5:
        data3.append(item[0])
        data4.append(item[1])
    cur2.close()
    con2.close()
    return render_template("flot.html",data3= data3,data4 =data4)

@app.route('/morris')
def morris():
    return render_template("morris.html")
@app.route('/tables')
def tables():
    datalist = []
    con1 = sqlite3.connect('C:\\Users\\86173\\Desktop\\毕设\\数据处理\\故障数据\\save_data')
    cur1 = con1.cursor()
    sql1 = "select * from data_test"
    data = cur1.execute(sql1)
    for item in data:
        datalist.append(item)
    cur1.close()
    con1.close()
    return render_template("tables.html",datalist = datalist)

@app.route('/blank')
def blank():
    return render_template("blank.html")

@app.route('/')
def zhuye():
    return render_template("zhuye.html")

@app.route("/predict_from_input", methods=["POST"])
def predict_from_input():
    # 从请求中获取数据并进行预测
    feature1 = float(request.form.get("feature1"))
    feature2 = float(request.form.get("feature2"))
    feature3 = float(request.form.get("feature3"))
    feature4 = float(request.form.get("feature4"))
    feature5 = float(request.form.get("feature5"))
    data = [[feature1, feature2, feature3, feature4,feature5]]
    prediction = model.predict(data)
    print(f'web上赋值的特征结果为：{prediction}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_high_dan.json", "w") as f:
        json.dump(prediction.tolist(), f)
    return jsonify(prediction=prediction.tolist())
@app.route("/predict_from_file", methods=["POST"])
def predict_from_file():
    # 从请求中获取文件并进行预测
    file = request.files.get("file")
    data = pd.read_csv(file)
    predictions = model.predict(data)
    print(f'本地上传的数据集结果为：{predictions}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_csv_high_dan.json", "w") as f:
        json.dump(predictions.tolist(), f)
    return jsonify(predictions=predictions.tolist())

@app.route("/predict_from_input2", methods=["POST"])
def predict_from_input2():
    # 从请求中获取数据并进行预测
    feature12 = float(request.form.get("feature12"))
    feature22 = float(request.form.get("feature22"))
    feature32 = float(request.form.get("feature32"))
    feature42 = float(request.form.get("feature42"))
    feature52 = float(request.form.get("feature52"))
    feature62 = float(request.form.get("feature62"))
    data = [[feature12, feature22, feature32, feature42, feature52, feature62]]
    prediction = model1.predict(data)
    print(f'web上赋值的特征结果为：{prediction}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_med_dan.json", "w") as f:
        json.dump(prediction.tolist(), f)
    return jsonify(prediction=prediction.tolist())
@app.route("/predict_from_file2", methods=["POST"])
def predict_from_file2():
    # 从请求中获取文件并进行预测
    file1 = request.files.get("file1")
    data1 = pd.read_csv(file1)
    predictions = model1.predict(data1)
    print(f'本地上传的数据集结果为：{predictions}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_csv_med_dan.json", "w") as f:
        json.dump(predictions.tolist(), f)
    return jsonify(predictions=predictions.tolist())

@app.route("/predict_from_input3", methods=["POST"])
def predict_from_input3():
    # 从请求中获取数据并进行预测
    feature13 = float(request.form.get("feature13"))
    feature23 = float(request.form.get("feature23"))
    feature33 = float(request.form.get("feature33"))
    feature43 = float(request.form.get("feature43"))
    data = [[feature13, feature23, feature33, feature43]]
    prediction = model2.predict(data)
    print(f'web上赋值的特征结果为：{prediction}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_low_dan.json", "w") as f:
        json.dump(prediction.tolist(), f)
    return jsonify(prediction=prediction.tolist())
@app.route("/predict_from_file3", methods=["POST"])
def predict_from_file3():
    # 从请求中获取文件并进行预测
    file2 = request.files.get("file2")
    data2 = pd.read_csv(file2)
    predictions = model2.predict(data2)
    print(f'本地上传的数据集结果为：{predictions}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_csv_low_dan.json", "w") as f:
        json.dump(predictions.tolist(), f)
    return jsonify(predictions=predictions.tolist())

@app.route("/predict_from_low", methods=["POST"])
def predict_from_low():
    # 从请求中获取数据并进行预测
    niuju = float(request.form.get("niuju"))
    gonglv = float(request.form.get("gonglv"))
    shichang = float(request.form.get("shichang"))
    zhuansu = float(request.form.get("zhuansu"))
    data = [[niuju, gonglv, shichang, zhuansu]]
    prediction = model3.predict(data)
    print(f'web上赋值的特征结果为：{prediction}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_input_low.json", "w") as f:
        json.dump(prediction.tolist(), f)
    return jsonify(prediction=prediction.tolist())
@app.route("/predict_from_lowfile", methods=["POST"])
def predict_from_lowfile():
    # 从请求中获取文件并进行预测
    file = request.files.get("file")
    data = pd.read_csv(file)
    predictions = model3.predict(data)
    print(f'本地上传的数据集结果为：{predictions}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_csv_low.json", "w") as f:
        json.dump(predictions.tolist(), f)
    return jsonify(predictions=predictions.tolist())

@app.route("/predict_from_med", methods=["POST"])
def predict_from_med():
    # 从请求中获取数据并进行预测
    shiwen = float(request.form.get("shiwen"))
    jiwen = float(request.form.get("jiwen"))
    zhuansu1 = float(request.form.get("zhuansu1"))
    niuju1 = float(request.form.get("niuju1"))
    gonglv1 = float(request.form.get("gonglv1"))
    shichang1 = float(request.form.get("shichang1"))
    data = [[shiwen, jiwen, zhuansu1, niuju1, gonglv1, shichang1]]
    prediction = model4.predict(data)
    print(f'web上赋值的特征结果为：{prediction}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_input_med.json", "w") as f:
        json.dump(prediction.tolist(), f)
    return jsonify(prediction=prediction.tolist())
@app.route("/predict_from_medfile", methods=["POST"])
def predict_from_medfile():
    # 从请求中获取文件并进行预测
    file = request.files.get("file")
    data = pd.read_csv(file)
    predictions = model4.predict(data)
    print(f'本地上传的数据集结果为：{predictions}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_csv_med.json", "w") as f:
        json.dump(predictions.tolist(), f)
    return jsonify(predictions=predictions.tolist())

@app.route("/predict_from_high", methods=["POST"])
def predict_from_high():
    # 从请求中获取数据并进行预测
    shiwen2 = float(request.form.get("shiwen2"))
    jiwen2 = float(request.form.get("jiwen2"))
    zhuansu2 = float(request.form.get("zhuansu2"))
    niuju2 = float(request.form.get("niuju2"))
    gonglv2 = float(request.form.get("gonglv2"))
    data = [[shiwen2, jiwen2, zhuansu2, niuju2, gonglv2]]
    prediction = model5.predict(data)
    print(f'web上赋值的特征结果为：{prediction}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_input_high.json", "w") as f:
        json.dump(prediction.tolist(), f)
    return jsonify(prediction=prediction.tolist())
@app.route("/predict_from_highfile", methods=["POST"])
def predict_from_highfile():
    # 从请求中获取文件并进行预测
    file = request.files.get("file")
    data = pd.read_csv(file)
    predictions = model5.predict(data)
    print(f'本地上传的数据集结果为：{predictions}')
    # 将预测结果保存为 JSON 文件
    with open("C:\\Users\\86173\\Desktop\\预测文件\\prediction_csv_high.json", "w") as f:
        json.dump(predictions.tolist(), f)
    return jsonify(predictions=predictions.tolist())

if __name__ == '__main__':
    app.run(debug = True)
