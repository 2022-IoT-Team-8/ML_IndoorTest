import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from sklearn import tree
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR, SVC
import joblib
import os



if __name__ == '__main__':

    print(os.getcwd())
    SVC(gamma='auto')
    train = pd.read_csv("./input/TrainingData.csv")
    test = pd.read_csv("./input/ValidationData.csv")

    drop_c = ['LONGITUDE', 'LATITUDE', 'RELATIVEPOSITION', 'BUILDINGID', 'RELATIVEPOSITION',
              'USERID', 'USERID', 'TIMESTAMP', 'PHONEID']

    drop_d = ['LONGITUDE', 'LATITUDE', 'RELATIVEPOSITION', 'SPACEID', 'BUILDINGID', 'RELATIVEPOSITION',
              'USERID', 'USERID', 'TIMESTAMP', 'PHONEID']
    train_c = (train.drop(drop_c, axis=1)).values
    test_c = (test.drop(drop_c, axis=1)).values

    # train_c =
    # train_d =


    X_train = train_c[:10000, :-1]
    Y_train = train_c[:10000, -1]

    X_test = train_c[10000:12000, :-1]
    Y_test = train_c[10000:12000, -1]

    rf_clf = RandomForestClassifier(random_state=0)
    rf_clf.fit(X_train, Y_train)

    print(rf_clf.predict(X_test))

    prdt_Y = []
    prdt_Y = rf_clf.predict(X_test)

    save = pickle.dumps(rf_clf)
    joblib.dump(rf_clf,"test_save.pkl")
    rf_clf = joblib.load("test_save.pkl")

    prdt_Y =rf_clf.predict(X_test)


    print(sklearn.metrics.mean_squared_error(Y_test, prdt_Y))
    print(sklearn.metrics.accuracy_score(Y_test, prdt_Y))
    print(test_c)
    print(prdt_Y)



