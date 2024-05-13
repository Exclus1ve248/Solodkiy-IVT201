import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

dataset = pd.read_excel('datasetTG.xlsx')
Y = dataset.iloc[:, -1]
X = dataset.iloc[:, :-1]
X = np.array(X)
Y = np.array(Y)
nX = preprocessing.normalize(X)
nY = preprocessing.normalize([Y])
model = LinearRegression()
model.fit(X,Y)
y_true1 = model.iloc[1]
y_pred1 = model.iloc[0]
mean_squared_error(y_true1, y_pred1)
with open('lin_reg', 'wb') as pkl:
    pickle.dump(model, pkl)