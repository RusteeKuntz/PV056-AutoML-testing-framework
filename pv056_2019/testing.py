import sklearn.feature_selection as sklfs
from pydantic import BaseModel
import pandas as pd
import numpy as np


class CommandSchema(BaseModel):
    """ This represents the schema of configuration for an arbitrary WEKA class invokation from command line """
    name: str
    parameters: dict = {}


def setup_sklearn_score_func(score_func: CommandSchema):
    # here we retrieve correct score function for FS by its name and set it up with parameters from JSON
    # TODO xbajger: We are not checking if the keyword arguments are right. We let the whole framework fail on an error
    #try:
    score_func = lambda x, y: getattr(sklfs, score_func.name)(x, y, **score_func.parameters)
    # if we got an unrecognized keyword, solve it by passing default arguments f
    #except TypeError:
    return score_func


def setup_sklearn_fs_class(class_schema: CommandSchema):
    # here we make use of a structural similarity between FS classes in scikit-learn
    # they all contain a "score_func" callable argument and then some other configuration
    if "score_func" not in class_schema.parameters:
        score_func = sklfs.chi2
    else:
        score_func = setup_sklearn_score_func(CommandSchema(**class_schema.parameters))
    # load class by name and construct instance with keyword arguments
    fsl = getattr(sklfs, class_schema.name)(score_func=score_func, **class_schema.parameters)
    return fsl

def select_features_with_sklearn(self, dataframe_without_classes: pd.DataFrame, classes: np.array)->pd.DataFrame:
    colnames = dataframe_without_classes.columns
    # if the k is negative, use it as "remove k worst" instead of "select k best" features.
    if "k" in self.settings["parameters"] and self.settings["parameters"]["k"] < 0:
        self.settings["parameters"]["k"] = len(colnames) + self.settings["parameters"]["k"]
    # split data and classes
    #x = dataframe_without_classes[colnames[:-1]]
    #y = dataframe_without_classes[colnames[-1]]
    # another score functions are: f_classif, mutual_info_classif
    if "score_func" not in self.settings:
        score_func=sklfs.chi2
    else:
        score_func = setup_sklearn_score_func(CommandSchema(**self.settings['score_func']))

    fs: sklfs.SelectKBest = sklfs.SelectKBest(score_func=score_func, **self.settings['parameters'])
    fs.fit(dataframe_without_classes, classes)
    transformed_df = fs.transform(dataframe_without_classes)
    return pd.DataFrame()


def main():
    df = pd.DataFrame([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    ], columns=["a", "b", "c", "d", "e"])

    colnames = df.columns
    x = df[colnames[:-1]]
    y = df[colnames[-1]]

    print(x.columns)

