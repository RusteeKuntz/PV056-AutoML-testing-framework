import sklearn.feature_selection as sklfs
from pydantic import BaseModel
import pandas as pd
import numpy as np
import resource
from sklearn.feature_selection.univariate_selection import _BaseFilter
from sklearn.preprocessing import OneHotEncoder

from pv056_2019.data_loader import DataLoader, DataFrameArff
from pv056_2019.schemas import ScikitFSSchema, FSStepSchema, CommandSchema

from pv056_2019.utils import SCIKIT, ID_NAME, WEKA_DATA_TYPES


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

def convert_multiindex_to_index(mi: pd.MultiIndex) -> [str]:
    # setup dictionary that will contain column names of column created by binarisation in lists under keys by their
    # original columns names before binarisation
    columns = {}
    for i in range(len(mi)):
        original_colname = mi.levels[0][mi.codes[0][i]]  # this extracts the original name of column
        catname = mi.levels[1][mi.codes[1][i]]  # this extracts the subcolumn names (categories)
        print(i, "|", original_colname, "|", catname)
        if original_colname not in columns:
            # create an entry for a column name
            columns[original_colname] = []
        # append to the list of subcolumn names (categories) of the column
        columns[original_colname].append(catname)
    print("DUH")
    print(columns)
    print(mi)
    print(mi.levels[0])

    #init a list of new columns
    new_columns = []
    for original_colname in columns.keys():
        #original_colname = mi.levels[0][mi.codes[0][i]]  # this again extracts the original name of column
        # the binarisation leaves names of WEKA data types instead of nominal values for columns representing
        # non-categorical values. To avoid  unnecessary renaming, we actually check for those specific names.
        if len(columns[original_colname]) == 1 and columns[original_colname][0] in WEKA_DATA_TYPES:
            new_columns.append(original_colname)
            #print(i, "|", original_colname)
        else:
            for subcolname in columns[original_colname]:
                new_columns.append(original_colname + "_" + subcolname)
                #print(i, "|", original_colname, "|", subcolname)

    print("NEW COLUMNS", new_columns)
    return new_columns


def select_features_with_sklearn(self, selector: _BaseFilter, leave_binarized: bool):
    colnames = self.columns
    # print(colnames)

    # make sure that ID column does not compromise the feature selection
    if ID_NAME in colnames:
        self.drop(ID_NAME)
    bin_df: pd.DataFrame = self._binarize_categorical_values()
    print("Original columns MultiIndex")
    print(bin_df.columns)

    # split data and classes. We rely on the fact that classes are in the last column
    x = bin_df.loc[:, colnames[:-1]]
    y = self.loc[:, colnames[-1]]

    # print(y)
    # another score functions are: f_classif, mutual_info_classif

    time_start = resource.getrusage(resource.RUSAGE_SELF)[0]
    time_start_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

    # fit selector to the dataset (this basically looks at the dataset and identifies useful features)
    selector.fit(x, y)

    selected_features_indexes = selector.get_support()
    # print(selected_features_indexes)

    # here we are indexing the binarized, filtered dataset by a list of bools.
    transformed_df: pd.DataFrame = x.iloc[:, selected_features_indexes]

    # print(transformed_df)
    nmi: pd.MultiIndex = transformed_df.columns

    if not leave_binarized:
        selected_feature_indexes_set = set()
        selected_feature_indexes_list = []
        for code in nmi.codes[0]:
            if code not in selected_feature_indexes_set:
                selected_feature_indexes_list.append(code)
            selected_feature_indexes_set.add(code)
        # here we actually add the "classes" column back into the list of selected features
        selected_feature_indexes_list.append(len(colnames) - 1)

        # print(selected_feature_indexes_list)
        final_df = self.iloc[:, selected_feature_indexes_list]

        # change the ARFF data so that they contain updated data
        selected_columns_set = set(final_df.columns)
        # obtain original arff data
        arff_data = self.arff_data()
        arff_data["attributes"] = [x for x in arff_data["attributes"] if x[0] in selected_columns_set]
        # adding the "arff_data" keyword bypasses the super.__init__() method in DataFrameArff, so we need to overwrite
        # the vlaues inside the arff_data themselves.
        arff_data["data"] = final_df.values
        new_frame_arff: DataFrameArff = DataFrameArff(arff_data=arff_data)

    else:
        new_columns = convert_multiindex_to_index(nmi)

        arff_data = {
            "relation": self._arff_data["relation"],
            "description": self._arff_data["description"],
            "attributes": [(name, 'NUMERIC') for name in new_columns],
            "data": transformed_df.values
        }
        new_frame_arff: DataFrameArff = DataFrameArff(arff_data=arff_data)

    # conclude time (resource) consumption
    time_end = resource.getrusage(resource.RUSAGE_SELF)[0]
    time_end_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

    fs_time = (time_end - time_start) + (time_end_children - time_start_children)

    return new_frame_arff, fs_time


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
        "source_library": "SCIKIT",
        "leave_attributes_binarized": True,
        "fs_method": {
            "name": "SelectFpr",
            "parameters": {
                "alpha": 0.8
            }
        },
        "score_func": {
            "name": "chi2",
            "parameters": {}
        }
    }

    fs_schema2 = {
        "source_library": SCIKIT,
        "leave_attributes_binarized": True,
        "fs_method": {
            "name": "SelectKBest",
            "parameters": {
                "k": 5
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

    if FSStepSchema(**fs_schema2).source_library == SCIKIT:
        conf = ScikitFSSchema(**fs_schema2)
        new_frame, fs_time = select_features_with_sklearn(df, setup_sklearn_fs_class(conf.fs_method, conf.score_func),
                                                          conf.leave_attributes_binarized)

        print(new_frame)
        new_frame.arff_dump("FUCKING_ARFF.arff")
