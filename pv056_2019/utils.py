# *********************************************************
# Utils for classifiers
# *********************************************************


ID_NAME = "ID"
OD_VALUE_NAME = "OD_VALUE"

# *********************************************************
# Utils for feature selection
# *********************************************************







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

class ArffData:
    relation: str
    description: str
    attributes: [(str, str)]
    data: [[any]]

    def __init__(self, relation: str, description: str, attributes: [(str, str)], data: [[any]]):
        self.relation = relation
        self.description = description
        self.attributes = attributes
        self.data = data

def valid_path(path, message):
    import os
    import argparse
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(message)
    else:
        return path


# df = pd.DataFrame([[1, 2, 2, 3, 4], [2, 1, 0, 0, 4], [0, 4, 15, 20, 9], [2, 3, 3, 11, 20]], columns=["a", "b", "c", "d", "e"])

