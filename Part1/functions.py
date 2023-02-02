import numpy as np
# from sklearn.x import y
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def tidyData(df):
    df = df.dropna(inplace=True)
    mean_house_price = df["SalePrice"].mean()
    df["SalePrice"] = np.where(df["SalePrice"] < mean_house_price, 0, 1)
    return df


def evaulateModel():
    pass


def fitLogisticRegression(df):
    # Separating features and target
    y = df["SalePrice"]
    X = df.drop(["SalePrice"], axis=1, inplace=True)

    # Make cross validation generator
    cv_generator = KFold(n_splits=10, shuffle=True, random_state=3)

    # initialise list for scores
    scores = []

    # make and scoe model on each fold
    for train_index, test_index in cv_generator.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #fit model
        clf = LogisticRegression(random_state=3).fit(X_train, y_train)

        #make prediction
        y_pred = clf.predict_proba(X_test)

        #score predition
        score = evaulateModel(y_test, y_pred)

        #add score to list
        scores.append(score)

    return scores


def fitNaiveBayes(df):
    pass
