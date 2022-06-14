import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from sklearn import tree
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
import joblib
import os

from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':

    print(os.getcwd())
    SVC(gamma='auto')
    train = pd.read_csv("./input/TrainingData.csv")
    test = pd.read_csv("./input/ValidationData.csv")


    # drop_c = ['longitude', 'latitude']
    # drop_d = ['longitude', 'latitude']
    drop_c = ['floor']
    drop_d = []

    train_c = (train.drop(drop_c, axis=1)).values
    test_c = (test.drop(drop_c, axis=1)).values

    X = train_c[:538, :-1]
    Y = train_c[:538, -1]
    # train_c =
    # train_d =




    # X_train = train_c[:538, :-1]
    # Y_train = train_c[:538, -1]
    #
    # X_test = train_c[500:538, :-1]
    # Y_test = train_c[500:538, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=None, random_state=14)



    # rf_clf = DecisionTreeClassifier()
    rf_clf = make_pipeline(StandardScaler(), tree.DecisionTreeClassifier())
    #rf_clf = RandomForestClassifier(random_state=17)
    rf_clf.fit(X_train, Y_train)

    #print(rf_clf.predict(X_test))

    prdt_Y = []
    prdt_Y = rf_clf.predict(X_test)

    save = pickle.dumps(rf_clf)
    joblib.dump(rf_clf, "test_save.pkl")
    rf_clf = joblib.load("test_save.pkl")

    prdt_Y =rf_clf.predict(X_test)


    #print(sklearn.metrics.mean_squared_error(Y_test, prdt_Y))
    print(sklearn.metrics.accuracy_score(Y_test, prdt_Y))

    #print(test_c)
    #print(prdt_Y)
