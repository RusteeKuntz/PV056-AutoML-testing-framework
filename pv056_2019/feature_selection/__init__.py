from __future__ import absolute_import

from typing import Any, Dict
import numpy as np
import pandas as pd
import sklearn.feature_selection as sklfs
from sklearn.feature_selection.univariate_selection import _BaseFilter

F_SELECTORS: Dict[str, Any] = {}


from pv056_2019.schemas import CommandSchema


class AbstractFeatureSelector:
    name: str
    data_type: str
    features: [str]

    def __init__(self, **settings):
        self.settings = settings

    def select_features(self, dataframe: pd.DataFrame, classes: np.array)->pd.DataFrame:
        raise NotImplementedError()





def feature_selector(cls):
    F_SELECTORS.update({cls.name: cls})
    return cls

# score_funcs: Dict[str, Any] = {
#     "chi2": sklfs.chi2,
#
# }
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



@feature_selector
class KBest(AbstractFeatureSelector):
    name = "KBEST"
    data_type = "REAL"

    def select_features(self, dataframe: pd.DataFrame, classes: np.array)->pd.DataFrame:
        colnames = dataframe.columns
        # if the k is negative, use it as "remove k worst" instead of "select k best" features.
        if "k" in self.settings["parameters"] and self.settings["parameters"]["k"] < 0:
            self.settings["parameters"]["k"] = len(colnames) + self.settings["parameters"]["k"]
        # split data and classes
        x = dataframe[colnames[:-1]]
        y = dataframe[colnames[-1]]
        # another score functions are: f_classif, mutual_info_classif
        if "score_func" not in self.settings:
            score_func=sklfs.chi2
        else:
            score_func = setup_sklearn_score_func(CommandSchema(**self.settings['score_func']))

        fs: sklfs.SelectKBest = sklfs.SelectKBest(score_func=score_func, **self.settings['parameters'])
        fs.fit(x, y)
        return fs.transform(x)



