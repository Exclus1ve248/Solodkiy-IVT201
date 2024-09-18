import pickle

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
def p_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Первый нейрон", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        predictions = new_neuron.forward(X_new)
        print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Помидор', 'Огурец'))
        return render_template('lab5.html', title="Первый нейрон", menu=menu,
                               class_model="Это: " + str(*np.where(predictions >= 0.5, 'Помидор', 'Огурец')))

if __name__ == "__main__":
    app.run(debug=True)
