# file for adding functions we wish to call in main report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

def trainLinearRegression(X, y):
    """
    X should be pandas dataframe of predictors (with to.dummies applied)
    y should be the saleprice column vector

    return is the mean absolute error of this model with leave one out
    cross validation applied to it.
    """
    abs_errors = []

    loo_generator = LeaveOneOut()
    reg = LinearRegression()

    for train_index, test_index in loo_generator.split(X):
        #subset dataframe
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #fit model
        reg.fit(X_train, y_train)

        #predict missing datapoint
        y_pred = reg.predict(X_test)

        #add absolute difference to list
        abs_errors.append(abs(y_test.to_numpy() - y_pred)[0])

    return np.sum(abs_errors)/len(abs_errors)


    