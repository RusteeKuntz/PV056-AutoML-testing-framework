# *********************************************************
# Utils for classifiers
# *********************************************************
import json
import re
from typing import List

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


def extract_parameter_value_as_int(json_string: str, parameter: str or List[str]):
    if isinstance(parameter, str):
        pattern = r"\s*[\"']?" + parameter + r"[\"']?\s*"
    elif isinstance(parameter, List) and len(parameter) > 1:
        pattern = "[\"']?(" + parameter[0]
        for par in parameter[1:]:
            pattern += r'|' + par
        pattern += ")[\"']?"
    elif isinstance(parameter, List) and len(parameter) == 1:
        pattern = r"\s*[\"']?" + parameter[0] + r"[\"']?\s*"
    else:
        print("Invalid parameter supplied. Only string or a list of strings is allowed. You supplied:",
              parameter, "of type", type(parameter))
        return json_string

    #print(pattern)
    extracted_value = re.findall(pattern + r':\s*[\'"]?(\d*|[\w.]*)["\']?[\s,}]', json_string)

    if isinstance(extracted_value, list):
        if len(extracted_value) == 0:
            #print("Did not find any matches for:", pattern + r':\s*["\']?([\d ]*|[\w\s]*)["\']?[\s,}]', "in string:", json_string)
            return json_string
        else:
            values = []
            for e in extracted_value:
                val = ""
                if isinstance(e, tuple):
                    # print("found one value")
                    try:
                        val = int(e[1])
                    except ValueError:
                        val = e[1]
                else:
                    try:
                        val = int(e)
                    except ValueError:
                        val = e
                if len(val) == 0:
                    values.append("UNMATCHED")
                else:
                    values.append(val)
            return ",".join(values)

    elif isinstance(extracted_value, str):
        #print("found one value")
        try:
            return int(extracted_value[0])
        except ValueError:
            return extracted_value[0]
    else:
        print(json_string, extracted_value, type(extracted_value))
        raise Exception("FATAL")
        # try:
        #     return ",".join([(e[1].split(".")[-1] if isinstance(e, tuple) else e.split(".")[-1] if len(e) > 0 else "UNMATCHED or EMPTY!") for e in extracted_value ])
        # except Exception as e:
        #
        #     raise e

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