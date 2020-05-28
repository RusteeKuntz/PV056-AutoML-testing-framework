# *********************************************************
# Utils for classifiers
# *********************************************************


ID_NAME = "ID"
OD_VALUE_NAME = "OD_VALUE"

# *********************************************************
# Utils for feature selection
# *********************************************************

import pandas as pd

def convert_multiindex_to_index(mi: pd.MultiIndex) -> [str]:
    # setup dictionary that will contain column names of column created by binarisation in lists under keys by their
    # original columns names before binarisation
    columns = {}
    for i in range(len(mi)):
        original_colname = mi.levels[0][mi.codes[0][i]]  # this extracts the original name of column
        catname = mi.levels[1][mi.codes[1][i]]  # this extracts the subcolumn names (categories)
        if original_colname not in columns:
            # create an entry for a column name
            columns[original_colname] = []
        # append to the list of subcolumn names (categories) of the column
        columns[original_colname].append(catname)

    #init a list of new columns
    new_columns = []
    for original_colname in columns.keys():
        # the binarisation leaves names of WEKA data types instead of nominal values for columns representing
        # non-categorical values. To avoid  unnecessary renaming, we actually check for those specific names.
        if len(columns[original_colname]) == 1 and columns[original_colname][0] in WEKA_DATA_TYPES:
            new_columns.append(original_colname)
        else:
            for subcolname in columns[original_colname]:
                new_columns.append(original_colname + "_" + subcolname)
    return new_columns



# *********************************************************
# Other utils
# *********************************************************

BASELINE_NAME = "baseline"
NONE_STR = "none"

# library identifiers
CUSTOM = "CUSTOM"
WEKA = "WEKA"
SCIKIT = "SCIKIT"

WEKA_DATA_TYPES = {"NUMERIC"}


def valid_path(path, message):
    import os
    import argparse
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(message)
    else:
        return path


# df = pd.DataFrame([[1, 2, 2, 3, 4], [2, 1, 0, 0, 4], [0, 4, 15, 20, 9], [2, 3, 3, 11, 20]], columns=["a", "b", "c", "d", "e"])

