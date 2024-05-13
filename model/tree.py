import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


fruit = pd.read_excel('fruit_tree.xlsx')
X=fruit.drop(["Fruit"],axis=1)
y=fruit["Fruit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)
with open('tree', 'wb') as pkl:
    pickle.dump(model, pkl)