import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


fruit_df=pd.read_csv("fruit (1).csv")
X=fruit_df.drop(["Fruit"],axis=1)
Y=fruit_df["Fruit"]
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.2,random_state=5)
model=LogisticRegression()
model.fit(X_train1,Y_train1)

with open('log_reg', 'wb') as pkl:
    pickle.dump(model, pkl)