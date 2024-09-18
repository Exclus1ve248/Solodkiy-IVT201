import numpy as np
from neuron import SingleNeuron


# Пример данных (X - входные данные, y - целевые значения)
X = np.array([  [-2, -1, -4],
                [25, 6, 4],
                [17, 4, 2],
                [-15, -6, -6]])
y = np.array([1, 0, 0, 1])  # Ожидаемый выход
# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3)
neuron.train(X, y, epochs=1000, learning_rate=0.1)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')