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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from custom_encoder import CategoricalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import hashlib

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


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[-in_test_set], data.loc[in_test_set]


if __name__ == "__main__":
    # fetch_housing_data()

    housing = load_housing_data()
    # print(housing.head())
    # print(housing.info())
    # print(housing["ocean_proximity"].value_counts())
    # print(housing.describe())
    # housing.hist(bins=50, figsize=(20, 15))
    # plt.show()

    # housing_with_id = housing.reset_index()
    # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

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
        housing_num = housing.drop("ocean_proximity", axis=1)
        # imputer.fit(housing_num)
        # print(imputer.statistics_)
        # print(housing_num.median().values)
        # X = imputer.transform(housing_num) # 用中位数替换空数据，结果为普通Numpy数组
        # housing_tr = pd.DataFrame(X, columns=housing_num.columns) # 转成Pandas DataFrame

        # 处理文本和类别属性
        encoder = LabelEncoder()
        housing_cat = housing["ocean_proximity"]
        housing_cat_encoded = encoder.fit_transform(housing_cat)
        print(housing_cat_encoded)
        print(encoder.classes_)
        #
        # encoder = OneHotEncoder()
        # housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
        # print(housing_cat_1hot)
        # print(housing_cat_1hot.toarray())

        # 自定义转换器
        # attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
        # housing_extra_attribs = attr_adder.transform(housing.values)

        # 转换流水线1
        # num_pipeline = Pipeline([
        #     ("imputer", Imputer(strategy="median")),
        #     ("attribs_adder", CombinedAttributesAdder()),
        #     ("std_scaler", StandardScaler()),
        # ])
        # housing_num_tr = num_pipeline.fit_transform(housing_num)
        # print(housing_num_tr)

        # 转换流水线2
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        num_pipeline = Pipeline([
            ("selector", DataFrameSelector(num_attribs)),
            ("imputer", Imputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ])
        cat_pipeline = Pipeline([
            ("selector", DataFrameSelector(cat_attribs)),
            ("cat_encoder", CategoricalEncoder(encoding="onehot-dense"))
        ])
        full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ])
        housing_prepared = full_pipeline.fit_transform(housing)
        # print(house_prepared)
        # print(house_prepared.shape)

        # 在训练集上训练和评估
        # 线性回归
        from sklearn.linear_model import LinearRegression

        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)

        some_data = housing.iloc[:5]
        some_labels = housing_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)
        print("Predictions:\n", lin_reg.predict(housing_prepared))
        print("Labels:\n", list(some_labels))

        from sklearn.metrics import mean_squared_error

        housing_predictions = lin_reg.predict(housing_prepared)
        lin_mse = mean_squared_error(housing_labels, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)
        print(lin_rmse)

        # 决策树回归
        from sklearn.tree import DecisionTreeRegressor

        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(housing_prepared, housing_labels)

        housing_predictions = tree_reg.predict(housing_prepared)
        tree_mse = mean_squared_error(housing_labels, housing_predictions)
        tree_rmse = np.sqrt(tree_mse)
        print(tree_rmse)

        # 使用交叉验证做评估
        # 决策树
        # scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
        #                          scoring="neg_mean_squared_error", cv=10)
        # tree_rmse_scores = np.sqrt(-scores)
        # display_scores(tree_rmse_scores)

        # 线性回归
        # lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
        #                              scoring="neg_mean_squared_error", cv=10)
        # lin_rmse_scores = np.sqrt(-lin_scores)
        # display_scores(lin_rmse_scores)

        # 随机森林
        # forest_reg = RandomForestRegressor(random_state=42)
        # forest_reg.fit(housing_prepared, housing_labels)
        # housing_predictions = forest_reg.predict(housing_prepared)
        # forest_mse = mean_squared_error(housing_labels, housing_predictions)
        # forest_rmse = np.sqrt(forest_mse)
        # print(forest_rmse)
        # forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
        #                                 scoring="neg_mean_squared_error", cv=10)
        # forest_rmse_scores = np.sqrt(-forest_scores)
        # display_scores(forest_rmse_scores)

        # 模型微调
        param_grid = [
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                   scoring="neg_mean_squared_error", return_train_score=True)
        grid_search.fit(housing_prepared, housing_labels)

        print(grid_search.best_params_)
        print(grid_search.best_estimator_)
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        # 分析最佳模型和它们的误差
        feature_importance = grid_search.best_estimator_.feature_importances_
        print(feature_importance)

        extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
        cat_one_hot_attribs = list(encoder.classes_)
        attributes = num_attribs + extra_attribs + cat_one_hot_attribs
        print(sorted(zip(feature_importance, attributes), reverse=True))

        # 用测试集评估系统
        final_model = grid_search.best_estimator_

        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()

        X_test_prepared = full_pipeline.transform(X_test)

        final_prediction = final_model.predict(X_test_prepared)

        final_mse = mean_squared_error(y_test, final_prediction)
        final_rmse = np.sqrt(final_mse)
        print(final_rmse)
