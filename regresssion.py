import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report


def line():
    print("===============================================================================")


if __name__ == '__main__':
    claimants = pd.read_csv("claimants.csv")

    print(claimants.head(10))
    line()

    # Drop the Case Number Column
    claimants.drop(["CASENUM"], inplace=True, axis=1)
    print(claimants.head(10))
    line()

    print("\nOVERALL DESCRIPTION OF DATA : \n")
    print(claimants.describe())
    line()

    print("UNIQUE VALUES IN ATTORNEY COLUMN : ")
    print(claimants["ATTORNEY"].unique())
    line()

    print("COUNT OF UNIQUE VALUES IN ATTORNEY COLUMN")
    print(claimants["ATTORNEY"].value_counts())
    line()

    print("The COLUMNS OF DATA : \n")
    print(claimants.columns)
    line()

    # plt.bar(claimants["ATTORNEY"], claimants["LOSS"])
    # plt.xlabel("ATTORNEY")
    # plt.ylabel("LOSS")
    # plt.show()

    print("NUMBER OF NULL VALUES IN DATA : ")
    print(claimants.isnull().sum())
    line()

    print("THE SHAPE OF DATA : ")
    print(claimants.shape)
    line()

    # FILLING IN NULL VALUES

    claimants["CLMSEX"].fillna(claimants.CLMSEX.mode(), inplace=True)
    claimants["CLMINSUR"].fillna(claimants.CLMINSUR.mode(), inplace=True)
    claimants["SEATBELT"].fillna(claimants.SEATBELT.mode(), inplace=True)
    claimants["CLMAGE"].fillna(claimants.CLMAGE.mode(), inplace=True)

    claimants.CLMSEX.fillna(claimants.CLMSEX.mean(), inplace=True)
    claimants.CLMINSUR.fillna(claimants.CLMINSUR.mean(), inplace=True)
    claimants.SEATBELT.fillna(claimants.SEATBELT.mean(), inplace=True)
    claimants.CLMAGE.fillna(claimants.CLMAGE.mean(), inplace=True)

    print("NUMBER OF NULL VALUES IN DATA AFTER FILLING IN DATA: ")
    print(claimants.isnull().sum())
    line()

    X = claimants.iloc[:, [1, 2, 3, 4, 5]]
    Y = claimants.iloc[:, 0]

    log_model = sm.Logit(Y, X).fit()
    print(log_model.summary())
    line()

    classifier = LogisticRegression()

    trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.8)
    classifier.fit(trainX, trainY)

    y_pred = classifier.predict(testX)

    print("THE TEST VALUES : ")
    print(testX)
    line(

    )
    print("THE PREDICTED VALUE FOR THE CHANCE OF HAVING AN ATTORNEY: ")
    print(y_pred)
