# file for adding functions we wish to call in main report
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
simplefilter("ignore", category=ConvergenceWarning)


def getUseableHPData():
    """
    Read and modify first houseprice dataframe
    to allow model fitting
    """
    # read data
    df = pd.read_csv("houseprices.csv")

    # preproccessing
    garage_types = df["GarageType"].tolist()
    fixed_garage_types = []

    for i in garage_types:
        if type(i) != str:
            fixed_garage_types.append("NoGarage")
        else:
            fixed_garage_types.append(i)

    df["GarageType"] = fixed_garage_types

    df.dropna(inplace=True)
    df = pd.get_dummies(df)
    return df


def getUseableHP2Data():
    """
    Read and modify second houseprices dataframe to allow for model fitting.
    This includes handling Na values and converting the dataframe to categorical.
    """
    # read data
    df = pd.read_csv("houseprices2.csv")

    # get all column data types, create dict from colname to dtype
    datatypes = [x for x in df.dtypes]
    colnames = df.dtypes.index
    dict_colname_to_dtype = dict(zip(colnames, datatypes))

    # get cunts of na values per column
    na_counts = [x for x in df.isna().sum()]

    # handle na vals
    for i in range(len(colnames)):
        column_name = colnames[i]
        column_type = dict_colname_to_dtype.get(column_name)
        column_num_na = na_counts[i]

    # for categorical columns, create na category
        if column_num_na > 10 and column_type == "object":
            categotical_column = df[column_name]
            fixed_column = np.where(
                categotical_column.isna(), "NotApplicable", categotical_column)
            df[column_name] = fixed_column

    # drop any non categorical where more than 10% of data missing
        if column_num_na > 150 and column_type != "object":
            df.drop(column_name, axis=1, inplace=True)

    df.dropna(inplace=True)
    df = pd.get_dummies(df)
    return df


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
        # subset dataframe
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # fit model
        reg.fit(X_train, y_train)

        # predict missing datapoint
        y_pred = reg.predict(X_test)

        # add absolute difference to list
        abs_errors.append(abs(y_test.to_numpy() - y_pred)[0])

    return np.sum(abs_errors)/len(abs_errors)


def trainLassoLinearRegression(X, y):
    """
    X should be pandas dataframe of predictors (with to.dummies applied)
    y should be the saleprice column vector

    return is the mean absolute error of this model with leave one out
    cross validation applied to it.
    """

    abs_errors = []

    loo_generator = LeaveOneOut()
    reg = Lasso(alpha=0.2, max_iter=25000)

    for train_index, test_index in loo_generator.split(X):
        # subset dataframe
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # fit model
        reg.fit(X_train, y_train)

        # predict missing datapoint
        y_pred = reg.predict(X_test)

        # add absolute difference to list
        abs_errors.append(abs(y_test.to_numpy() - y_pred)[0])

    return np.sum(abs_errors)/len(abs_errors)


def quickScoreLinearRegression(colnames, use_hp_2=False):
    """
    arguments_list contains a list of function arguments with the following:
    - queue: multiprocessing queue
    - y: the column vector we wish to predict
    - X: the pandas dataframe of predictors
    we wish to use in our model

    This funciton is built in this way to optimize for parallel processing of results.
    """
    # read and preprocess data
    if use_hp_2 == False:
        df = getUseableHPData()
    else:
        df = getUseableHP2Data()

    y = df["SalePrice"]
    X = df[colnames]

    abs_errors = []

    loo_generator = LeaveOneOut()
    reg = LinearRegression()

    for train_index, test_index in loo_generator.split(X):
        # subset dataframe
        if len(colnames) == 1:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # fit model
        reg.fit(X_train, y_train)

        # predict missing datapoint
        y_pred = reg.predict(X_test)

        # add absolute difference to list
        abs_errors.append(abs(y_test.to_numpy() - y_pred)[0])

    mae = np.sum(abs_errors)/len(abs_errors)
    return mae


def trainRandomForest(X, y):
    """
    arguments_list contains a list of function arguments with the following:
    - queue: multiprocessing queue
    - y: the column vector we wish to predict
    - X: the pandas dataframe of predictors
    we wish to use in our model

    This funciton is built in this way to optimize for parallel processing of results.
    """

    abs_errors = []

    cv_generator = KFold(n_splits=10, shuffle=True, random_state=3)
    rf = RandomForestRegressor(n_estimators=20, random_state=3)

    for train_index, test_index in cv_generator.split(X):
        # subset dataframe
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # fit model
        rf.fit(X_train, y_train)

        # predict missing datapoint
        y_pred = rf.predict(X_test)

        # add absolute difference to list
        abs_errors.append(abs(y_test.to_numpy() - y_pred)[0])

    mae = np.sum(abs_errors)/len(abs_errors)
    return mae
