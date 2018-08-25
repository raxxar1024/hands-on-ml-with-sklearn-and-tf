# -*- coding: utf-8 -*-
import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    # fetch_housing_data()

    housing = load_housing_data()
    # print(housing.head())
    # print(housing.info())
    # print(housing["ocean_proximity"].value_counts())
    # print(housing.describe())
    # housing.hist(bins=50, figsize=(20, 15))
    # plt.show()

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
        # print(housing["income_cat"].value_counts() / len(housing))
        #
        # for set in (strat_train_set, strat_test_set):
        #     set.drop(["income_cat"], axis=1, inplace=True)
        #
        housing = strat_train_set.copy()

        # housing.plot(kind="scatter", x="longitude", y="latitude")
        # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
        # plt.show()
        #
        # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        #              s=housing["population"] / 100, label="population",
        #              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
        # plt.legend()
        # plt.show()

        # 查找关联
        # corr_matrix = housing.corr()
        # print(corr_matrix["median_house_value"].sort_values(ascending=False))

        # 每个数值属性对每个其它数值属性的图
        # attributes = ["median_house_value", "median_income", "total_rooms",
        #               "housing_median_age"]
        # scatter_matrix(housing[attributes], figsize=(12, 8))
        # plt.show()

        # 最有希望用来预测房价中位数的属性是收入中位数
        # housing.plot(kind="scatter", x="median_income", y="median_house_value",
        #              alpha=0.1)
        # plt.show()

        # 创建属性
        housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
        housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
        housing["population_per_household"] = housing["population"]/housing["households"]
        corr_matrix = housing.corr()
        print(corr_matrix["median_house_value"].sort_values(ascending=False))
