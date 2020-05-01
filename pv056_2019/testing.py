import sklearn.feature_selection as sklfs
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_selection.univariate_selection import _BaseFilter

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


def select_features_with_sklearn(dataframe: pd.DataFrame, selector: _BaseFilter) -> pd.DataFrame:
    colnames = dataframe.columns

    # split data and classes. We rely on the fact that classes are in the last column
    x = dataframe[colnames[:-1]]
    y = dataframe[colnames[-1]]
    # another score functions are: f_classif, mutual_info_classif

    selector.fit(x, y)
    # remove features from the dataset without classes.
    # selector.transform(x)
    selected_features = selector.get_support()
    transformed_df = x.iloc[:, selected_features]

    # print(selected_features)
    # print(transformed_df)
    # print(type(transformed_df))

    # push classes back in
    transformed_df[colnames[-1]] = y
    return transformed_df


def main():
    df = pd.DataFrame([
        [1, 1, 2, 4, 8],
        [0, 1, 1, 3, 5],
        [1, 2, 3, 6, 13],
        [53, 0, 3, 7, 15],
        [1, 2, 1, 2, 3],
        [0, 1, 1, 3, 7],
        [53, 2, 4, 7, 15],
        [4, 0, 3, 6, 12]

    ], columns=["a", "b", "c", "d", "e"])

    fs_schema = {
        "source_library": SCIKIT,
        "fs_method": {
            "name": "SelectFpr",
            "parameters": {
                "alpha": 0.5
            }
        },
        "score_func": {
            "name": "mutual_info_classif",
            "parameters": {
                "n_neighbors": 4,
                "random_state": 123
            }

        }

    }

    # colnames = df.columns

    # df['kokot'] = pd.Series([8 for _ in range(8)])
    # df[colnames[-1]] = pd.Series([8 for _ in range(8)])
    # print(df)

    if FSStepSchema(**fs_schema).source_library == SCIKIT:
        conf = ScikitFSSchema(**fs_schema)
        newFrame = select_features_with_sklearn(df, setup_sklearn_fs_class(conf.fs_method, conf.score_func))

        print(newFrame)
