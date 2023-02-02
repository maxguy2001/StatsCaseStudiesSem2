import numpy as np
# from sklearn.x import y
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def tidyData(df):
    df.dropna(inplace=True)
    mean_house_price = df["SalePrice"].mean()
    df["SalePrice"] = np.where(df["SalePrice"] < mean_house_price, 0, 1)
    return df


def evaulateModel():
    pass


def logScore(real_values, predictions):

    log_score = 0
    for i in range(len(real_values)):
        log_score += np.log(predictions[i, real_values[i]])

    return log_score


def brierScore(real_values, predictions):
    predict_one = predictions[:, 1]
    squared_diffs = 0
    for i in range(len(real_values)):
        squared_diffs += (real_values[i] - squared_diffs[i])**2

    brier_score = squared_diffs/len(real_values)
    return brier_score


def fitLogisticRegression(df, scoringFunction):
    # encode categorical data as binary dummy variables
    df = pd.get_dummies(df)

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

        # scale data
        X_train = StandardScaler().fit_transofrm(X_train)
        X_test = StandardScaler().fit_transofrm(X_test)

        # fit model
        clf = LogisticRegression(random_state=3).fit(X_train, y_train)

        # make prediction
        y_pred = clf.predict_proba(X_test)

        # score predition
        score = scoringFunction(y_test, y_pred)

        # add score to list
        scores.append(score)

    return scores


def fitNaiveBayes(df):
    pass
