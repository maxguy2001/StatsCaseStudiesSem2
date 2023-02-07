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
        probability = predictions[i, real_values[i]]
        if abs(probability) < 0.00001:
            log_score -= np.log(0.00001)
        else:
            log_score -= np.log(probability)

    return log_score


def brierScore(real_values, predictions):
    predict_one = predictions[:, 1]
    squared_diffs = 0
    for i in range(len(real_values)):
        squared_diffs += (real_values[i] - predict_one[i])**2

    brier_score = squared_diffs/len(real_values)
    return brier_score


def fitLogisticRegression(df, scoringFunction):
    # encode categorical data as binary dummy variables
    df = pd.get_dummies(df)

    # Separating features and target
    y = df["SalePrice"]
    X = df.copy(deep=True)
    X.drop(["SalePrice"], axis=1, inplace=True)

    # Make cross validation generator
    cv_generator = KFold(n_splits=10, shuffle=True, random_state=3)

    # initialise list for scores
    scores = []

    # make model object
    clf = LogisticRegression(random_state=3, class_weight="balanced")

    # make and scoe model on each fold
    for train_index, test_index in cv_generator.split(X):

        # get testing and training data in fold
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # scale data
        X_train_scaled = StandardScaler().fit_transform(X_train)
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)

        X_test_scaled = StandardScaler().fit_transform(X_test)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

        # fit model
        clf.fit(X_train, y_train)

        # make prediction
        y_pred = clf.predict_proba(X_test)

        # score prediction
        score = scoringFunction(y_test.to_numpy(), y_pred)

        # add score to list
        scores.append(score)

    return scores


def fitNaiveBayes(df):
    pass
