import pickle

import tensorflow as tf
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify

from model.neuron import SingleNeuron

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_lab1"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"},
        {"name": "Лаба 5", "url": "p_lab5"}]

loaded_model_lin_reg = pickle.load(open('model/lin_reg', 'rb'))
loaded_model_log_reg = pickle.load(open('model/log_reg', 'rb'))
loaded_model_knn = pickle.load(open('model/knn', 'rb'))
loaded_model_tree = pickle.load(open('model/tree', 'rb'))
new_neuron = SingleNeuron(input_size=2)
new_neuron.load_weights('model/neuron_weights.txt')
model_class = tf.keras.models.load_model('model/classification_model.h5')

@app.route('/api', methods=['get'])
def get_weather():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['height']),
                       float(request_data['weight']),
                       float(request_data['gender'])]])
    pred = loaded_model_lin_reg.predict(X_new)

    return jsonify(shoe_size=pred[0])

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_lab1", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Линейная регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        pred = loaded_model_lin_reg.predict(X_new)
        return render_template('lab1.html', title="Линейная регрессия", menu=menu,
                               class_model=pred)

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        pred = loaded_model_log_reg.predict(X_new)
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model=pred)


@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Метод K-ближайших соседей kNN", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab3.html', title="Метод K-ближайших соседей kNN", menu=menu,
                               class_model=pred)

@app.route("/p_lab4", methods=['POST', 'GET'])
def f_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Дерево решений", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        pred = loaded_model_tree.predict(X_new)
        return render_template('lab3.html', title="Дерево решений", menu=menu,
                               class_model=pred)

@app.route("/p_lab5", methods=['POST', 'GET'])
def p_lab5():
    if request.method == 'GET':
        return render_template('lab5.html', title="Первый нейрон", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1'])-170,
                           float(request.form['list2'])-55,
                           float(request.form['list3'])-35]])
        predictions = new_neuron.forward(X_new)
        print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Женщина', 'Мужчина'))
        return render_template('lab5.html', title="Первый нейрон", menu=menu,
                               class_model="Это: " + str(*np.where(predictions >= 0.5, 'Женщина', 'Мужчина')))

@app.route('/api_class', methods=['get'])
def predict_classification():
    # Получение данных из запроса http://localhost:5000/api_class?height=-2&wight=-1&shoe=-4
    input_data = np.array([[float(request.args.get('height')),
                            float(request.args.get('wight')),
                            float(request.args.get('shoe'))
                            ]])
    print(input_data)
    predictions = model_class.predict(input_data)
    print(predictions)
    result = 'Женщина' if predictions >= 0.5 else 'Мужчина'
    print(result)
    app.config['JSON_AS_ASCII'] = False
    return jsonify(predict = str(result))

if __name__ == "__main__":
    app.run(debug=True)
