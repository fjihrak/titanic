import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def logreg_modeling(data, label, c):
    logreg = LogisticRegression(C=c).fit(data, label)

    #R^2
    #print('Training set score(R^2): {:.3f}'.format(logreg.score(data, label)))

    return logreg


def logreg_predict(model, data, prem):
    score = []
    y_pred = model.predict_proba(data)

    for i in range(len(y_pred)):
        y_pred[i][0] = round(y_pred[i][0], 3)
        y_pred[i][1] = round(y_pred[i][1], 3)
        score.append([prem[i][0],y_pred[i][1]])

    return score

def logreg_validation(model, data, prem, label):
    score = []
    y_pred = model.predict_proba(data)

    for i in range(len(y_pred)):
        y_pred[i][0] = round(y_pred[i][0], 3)
        y_pred[i][1] = round(y_pred[i][1], 3)
        score.append([prem[i][0],y_pred[i][1], label[i]])

    return score

def neural_network_model(data, label, size, layer):
    model = MLPClassifier(solver="adam", random_state=0, max_iter=10000, hidden_layer_sizes=(size, layer))
    model.fit(data, label)

    return model

def neural_network_predict(model, data, prem):
    y_pred = model.predict_proba(data)

    score =[]
    for i in range(len(y_pred)):
        y_pred[i][0] = round(y_pred[i][0], 3)
        y_pred[i][1] = round(y_pred[i][1], 3)
        score.append([prem[i][0],y_pred[i][1]])

    return score

def neural_network_validation(model, data, prem, label):
    y_pred = model.predict_proba(data)

    score = []
    for i in range(len(y_pred)):
        y_pred[i][0] = round(y_pred[i][0], 3)
        y_pred[i][1] = round(y_pred[i][1], 3)
        score.append([prem[i][0],y_pred[i][1],label[i]])

    return score

def gradientBoostingClassifier_model(data, label):
    gbrt = GradientBoostingClassifier(random_state=0, n_estimators=100, max_depth=5)
    gbrt.fit(data, label)

    return gbrt

def gradientBoostingClassifier_predict(model, data, prem):
    score = []
    y_pred = model.predict_proba(data)

    for i in range(len(y_pred)):
        y_pred[i][0] = round(y_pred[i][0], 3)
        y_pred[i][1] = round(y_pred[i][1], 3)
        score.append([prem[i][0],y_pred[i][1]])

    return score

def gradientBoostingClassifier_validation(model, data, prem, label):
    y_pred = model.predict_proba(data)

    score = []
    for i in range(len(y_pred)):
        y_pred[i][0] = round(y_pred[i][0], 3)
        y_pred[i][1] = round(y_pred[i][1], 3)
        score.append([prem[i][0],y_pred[i][1],label[i]])

    return score

def accuracy(score, label, threshold):
    cl = []
    for i in range(len(score)):
        if score[i][1] >= threshold:
            cl.append(1)
        else:
            cl.append(0)

    ac = 0
    for i in range(len(label)):
        if int(cl[i]) == int(label[i]):
            ac += 1

    accuracy = ac / len(label)
    print('accuracy(threshold:' + str(threshold) + ')' + str(round(accuracy,3)))
