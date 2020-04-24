import sklearn.feature_selection as sklfs
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_selection.univariate_selection import _BaseFilter


class CommandSchema(BaseModel):
    """ This represents the schema of configuration for an arbitrary WEKA class invokation from command line """
    name: str
    parameters: dict = {}


def setup_sklearn_score_func(score_func_schema: CommandSchema):
    # here we retrieve correct score function for FS by its name and set it up with parameters from JSON
    # TODO xbajger: We are not checking if the keyword arguments are right. We let the whole framework fail on an error
    #try:
    score_func = lambda x, y: getattr(sklfs, score_func_schema.name)(x, y, **score_func_schema.parameters)
    # if we got an unrecognized keyword, solve it by passing default arguments f
    #except TypeError:
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
    transformed_df = selector.transform(x)
    # push classes back in
    transformed_df[colnames[-1]] = y
    return transformed_df


def main():
    df = pd.DataFrame([
        [1, 2, 2, 4, 5],
        [2, 2, 1, 2, 3],
        [2, 1, 1, 2, 3],
        [1, 58, 0, 1, 2],
        [1, 2, 3, 5, 6],
        [2, 2, 20, 41, 42],
        [1, 2, 6, 11, 12],
        [1, 2, 5, 10, 11]
    ], columns=["a", "b", "c", "d", "e"])

    class_schema = CommandSchema(**{
        "name": "SelectKBest",
        "parameters": {
            "k": 4
        }
    })
    csf_schema = CommandSchema(**{
        "name": "f_classif",
        "parameters": {}
    })

    newFrame = select_features_with_sklearn(df, setup_sklearn_fs_class(class_schema, csf_schema))

    print(newFrame)

