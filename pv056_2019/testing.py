import sklearn.feature_selection as sklfs
from pydantic import BaseModel
import pandas as pd
import numpy as np
import resource
from sklearn.feature_selection.univariate_selection import _BaseFilter
from sklearn.preprocessing import OneHotEncoder

from pv056_2019.data_loader import DataLoader, DataFrameArff
from pv056_2019.schemas import ScikitFSSchema, FSStepSchema, CommandSchema

from pv056_2019.utils import SCIKIT


def setup_sklearn_score_func(score_func_schema: CommandSchema):
    # here we retrieve correct score function for FS by its name and set it up with parameters from JSON
    # TODO xbajger: We are not checking if the keyword arguments are right. We let the whole framework fail on an error
    # try:
    score_func = lambda x, y: getattr(sklfs, score_func_schema.name)(x, y, **score_func_schema.parameters)
    # if we got an unrecognized keyword, solve it by passing default arguments f
    # except TypeError:
    return score_func


def setup_sklearn_fs_class(class_schema: CommandSchema, score_func_schema: CommandSchema = None) -> _BaseFilter:
    # here we make use of a structural similarity between FS classes in scikit-learn
    # they all contain a "score_func" callable argument and then some other configuration
    # so we need 2 config: for class and for score_func
    if score_func_schema is None:
        score_func_schema = CommandSchema(**{"name": "chi2", "parameters": {}})

    score_func = setup_sklearn_score_func(score_func_schema)
    # load class by name and construct instance with keyword arguments
    fsl = getattr(sklfs, class_schema.name)(score_func=score_func, **class_schema.parameters)
    return fsl


def select_features_with_sklearn(self, selector: _BaseFilter):
    colnames = self.columns
    bin_df: pd.DataFrame = self._binarize_categorical_values()
    # Index(['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'Class'], dtype='object')
    # MultiIndex(levels=[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8'], ['F', 'I', 'M', 'NUMERIC']],
    #            codes=[[0, 0, 0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 3, 3, 3, 3, 3, 3]],
    #            names=['0', '1'])
    #mi: pd.MultiIndex = bin_df.columns  # We have a MUltiIndex on our hands here
    #original_columns = mi.levels[0]

    # split data and classes. We rely on the fact that classes are in the last column
    x = bin_df.loc[:, colnames[:-1]]
    # print(x.dtypes)
    # 0   1
    # V1  F       float64
    #     I       float64
    #     M       float64
    # V2  NUMERIC float64
    # V3  NUMERIC float64
    # V4  NUMERIC float64
    # V5  NUMERIC float64
    # V6  NUMERIC float64
    # V7  NUMERIC float64
    # V8  NUMERIC float64
    # dtype: object
    y = self.loc[:, colnames[-1]]

    # print(y)
    # another score functions are: f_classif, mutual_info_classif

    time_start = resource.getrusage(resource.RUSAGE_SELF)[0]
    time_start_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

    selector.fit(x, y)
    # remove features from the dataset without classes.
    # selector.transform(x)

    selected_features_indexes = selector.get_support()
    # print(selected_features_indexes)

    # here we are indexing by a list of bools.
    transformed_df = x.iloc[:, selected_features_indexes]
    print(transformed_df)
    nmi = transformed_df.columns

    selected_feature_indexes_set = set()
    selected_feature_indexes_list = []
    for code in nmi.codes[0]:
        if code not in selected_feature_indexes_set:
            selected_feature_indexes_list.append(code)
        selected_feature_indexes_set.add(code)

    print(selected_feature_indexes_list)
    final_df = self.iloc[:, selected_feature_indexes_list]
    print(final_df)

    # push classes back into the dataframe
    # transformed_df.loc[:, colnames[-1]] = y

    # create new ARFF dataframe object
    selected_columns_set = set(final_df.columns)
    arff_data = self.arff_data()
    arff_data["attributes"] = [x for x in arff_data["attributes"] if x[0] in selected_columns_set]

    print(arff_data["data"])
    new_frame_arff: DataFrameArff = DataFrameArff(final_df.values, arff_data=arff_data)
    #new_frame_arff._arff_data = self.arff_data()  # reassign full arff data

    # attribute_set = set(transformed_df.columns)  # create the set of selected attributes
    # # reassing arff data attribues and retain only those arff attributes that were selected,
    # # presuming that column names are same as arff attribute names
    # new_frame_arff._arff_data["attributes"] = [
    #     x
    #     for x in self._arff_data["attributes"]
    #     if x[0] in attribute_set
    # ]

    # conclude time (resource) consumption
    time_end = resource.getrusage(resource.RUSAGE_SELF)[0]
    time_end_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

    fs_time = (time_end - time_start) + (time_end_children - time_start_children)

    return new_frame_arff  , fs_time


def main():
    # df = pd.DataFrame([
    #     [1, 1, 2, 4, 8],
    #     [0, 1, 1, 3, 5],
    #     [1, 2, 3, 6, 13],
    #     [53, 0, 3, 7, 15],
    #     [1, 2, 1, 2, 3],
    #     [0, 1, 1, 3, 7],
    #     [53, 2, 4, 7, 15],
    #     [4, 0, 3, 6, 12]
    #
    # ], columns=["a", "b", "c", "d", "e"])

    df = DataLoader._load_arff_file("data/datasets/abalone.arff")

    fs_schema = {
        "source_library": SCIKIT,
        "fs_method": {
            "name": "SelectKBest",
            "parameters": {
                "k": 6
            }
        },
        "score_func": {
            "name": "chi2",
            "parameters": {}

        }

    }

    # colnames = df.columns

    # df['kokot'] = pd.Series([8 for _ in range(8)])
    # df[colnames[-1]] = pd.Series([8 for _ in range(8)])
    # print(df)

    if FSStepSchema(**fs_schema).source_library == SCIKIT:
        conf = ScikitFSSchema(**fs_schema)
        new_frame = select_features_with_sklearn(df, setup_sklearn_fs_class(conf.fs_method, conf.score_func))

        print(new_frame)
