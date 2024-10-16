import tensorflow as tf

import numpy as np

X_class = np.array([  [-2, -1, 0],
                [25, 6, 7],
                [17, 4, 6],
                [-15, -6, 1]])
y_class = np.array([1, 0, 0, 1])

# Создание модели для классификации
model_class = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Один выход для бинарной классификации
])

model_class.compile(optimizer='adam', loss='binary_crossentropy')

# Обучение модели
model_class.fit(X_class, y_class, epochs=100, batch_size=32)

# Прогноз
test_data = np.array([[-2,-1,-4]])
y_pred_class = model_class.predict(test_data)
print("Предсказанные значения:", y_pred_class, *np.where(y_pred_class >= 0.5, 'Женщина', 'Мужчина'))
# Сохранение модели для классификации
model_class.save('classification_model.h5')