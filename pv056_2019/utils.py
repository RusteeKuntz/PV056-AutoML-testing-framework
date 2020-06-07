# *********************************************************
# Utils for classifiers
# *********************************************************
import json
import re

ID_NAME = "ID"
OD_VALUE_NAME = "OD_VALUE"

# *********************************************************
# Utils for feature selection
# *********************************************************




# *********************************************************
# Utils for graph creation
# *********************************************************

def locate_string_in_arr(arr: [], string: str):
    index = 0
    while arr[index] != string:
        index+=1
    return index


def extract_parameter_value_as_int(json_string: str, parameter: str):
    extracted_value = re.search(parameter + r': (.*|\d*)(\s|,|})', json_string).group(1)
    try:
        return int(extracted_value)
    except ValueError:
        return extracted_value

def convert_dict_to_parameter_pairs(json_string: str):
    _dct = json.loads(json_string)
    res = ""
    first = True
    for key in _dct:
        if not first:
            res += ", "
        if first:
            first = not first
        res += key+"="+str(_dct[key])

    return res



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

    def __init__(self, relation: str, attributes: [(str, str)], description: str = "", data: [[any]] = None):
        if data is None:
            data = []
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

def _nest_quotes(string, which_quotes="\""):
    # TODO: escaping escape slashes themselves might not be necessary, check it later
    # string = re.sub(r"\\", r"\\\\", string)
    return re.sub(which_quotes, "\\" + which_quotes, string)


def _nest_double_quotes(string):
    return _nest_quotes(string, "\"")


def _nest_single_quotes(string):
    return _nest_quotes(string, "'")


def _assert_trailing_slash(string):
    if string[-1] != "/":
        return string + "/"
    else:
        return string