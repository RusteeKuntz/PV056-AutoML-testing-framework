# *********************************************************
# Utils for classifiers
# *********************************************************

ID_NAME = "ID"
OD_VALUE_NAME = "OD_VALUE"

# *********************************************************
# Other utils
# *********************************************************

BASELINE_NAME = "baseline"
NONE_STR = "none"

# library identifiers
CUSTOM = "CUSTOM"
WEKA = "WEKA"
SCIKIT = "SCIKIT"


def valid_path(path, message):
    import os
    import argparse
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(message)
    else:
        return path

# df = pd.DataFrame([[1, 2, 2, 3, 4], [2, 1, 0, 0, 4], [0, 4, 15, 20, 9], [2, 3, 3, 11, 20]], columns=["a", "b", "c", "d", "e"])