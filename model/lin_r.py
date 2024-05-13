import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

dataset = pd.read_excel('datasetTG.xlsx')
Y = dataset.iloc[:, -1]
X = dataset.iloc[:, :-1]
X = np.array(X)
Y = np.array(Y)
nX = preprocessing.normalize(X)
nY = preprocessing.normalize([Y])
model = LinearRegression()
with open('lin_reg', 'wb') as pkl:
    pickle.dump(model, pkl)