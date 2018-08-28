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
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

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


# 自定义转换器

room_ix, bedroom_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, room_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, room_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedroom_ix] / X[:, room_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


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
        # housing = strat_train_set.copy()

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
        # housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
        # housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
        # housing["population_per_household"] = housing["population"]/housing["households"]
        # corr_matrix = housing.corr()
        # print(corr_matrix["median_house_value"].sort_values(ascending=False))

        # 为机器学习算法准备数据
        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        # 数据清洗
        # imputer = Imputer(strategy="median")
        # housing_num = housing.drop("ocean_proximity", axis=1)
        # imputer.fit(housing_num)
        # print(imputer.statistics_)
        # print(housing_num.median().values)
        # X = imputer.transform(housing_num) # 用中位数替换空数据，结果为普通Numpy数组
        # housing_tr = pd.DataFrame(X, columns=housing_num.columns) # 转成Pandas DataFrame

        # 处理文本和类别属性
        # encoder = LabelEncoder()
        # housing_cat = housing["ocean_proximity"]
        # housing_cat_encoded = encoder.fit_transform(housing_cat)
        # print(housing_cat_encoded)
        # print(encoder.classes_)
        #
        # encoder = OneHotEncoder()
        # housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
        # print(housing_cat_1hot)
        # print(housing_cat_1hot.toarray())

        # 自定义转换器
        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
        housing_extra_attribs = attr_adder.transform(housing.values)
