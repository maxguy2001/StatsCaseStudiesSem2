import numpy as np
# from sklearn.x import y
import pandas as pd


def tidyData(df):
    df = df.dropna(inplace=True)
    mean_house_price = df["SalePrice"].mean()
    df["SalePrice"] = np.where(df["SalePrice"] < mean_house_price, 0, 1)
    return df


def evaulateModel():
    pass


def fitLogisticRegression(df):
    pass


def fitNaiveBayes(df):
    pass
