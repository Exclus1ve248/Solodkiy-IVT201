import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruit_df = pd.read_excel("fruit.xlsx")
X = fruit_df.drop(["Fruit"], axis=1)
Y = fruit_df["Fruit"]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.3, random_state=5)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train1, Y_train1)
with open('knn', 'wb') as pkl:
    pickle.dump(model, pkl)