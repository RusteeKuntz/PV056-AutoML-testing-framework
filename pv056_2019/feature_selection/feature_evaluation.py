import argparse
import hashlib
import re, os, json

import arff

from pv056_2019.schemas import WekaClassCommandSchema, FeatureSelectionStepSchema, \
    FeatureSelectionFilterConfigurationSchema
from pv056_2019.data_loader import DataLoader, DataFrameArff
from pv056_2019.utils import OD_VALUE_NAME


class FSCommandWithInfo:
    args: []
    dataset: str
    fold: str
    eval_method_name: str
    output_file_path: str

    def __init__(self, args: [str], ds: str, fold: str, eval: str, out: str) -> None:
        self.args = args
        self.dataset = ds
        self.fold = fold
        self.eval_method_name = eval
        self.output_file_path = out

    def __str__(self) -> str:
        return " ".join(self.args)


def _nest_quotes(string, which_quotes="\""):
    # TODO: escaping escape slashes themselves might not be necessary, check it later
    # string = re.sub(r"\\", r"\\\\", string)
    return re.sub(which_quotes, r"\\" + which_quotes, string)


def _nest_double_quotes(string):
    return _nest_quotes(string, "\"")


def _nest_single_quotes(string):
    return _nest_quotes(string, "'")


def _assert_trailing_slash(string):
    if string[-1] != "/":
        return string + "/"
    else:
        return string


def get_weka_command_from_config(config: WekaClassCommandSchema) -> str:
    """
    This will return a command containing an executable WEKA class with arguments, enclosed in double quotes,
    ready to be placed as a parameter to another executable WEKA class.
    """
    _command = config.class_name
    _command += get_weka_command_arguments_for_class(config.parameters)

    return _command


def get_weka_command_arguments_for_class(parameters: dict):
    _command = ""
    for param_name, param_val in parameters.items():
        if param_val:
            _command += " -" + param_name
            if not isinstance(param_val, bool):
                _command += " " + _nest_double_quotes(str(param_val))

    return _command


class FeatureSelectionManager:
    """ This class serves as a tool for building full java command for command line to execute in an external process
    example commandline input:

     java -cp data/java/weka.jar weka.attributeSelection.AttributeSelection weka.attributeSelection.InfoGainAttributeEval -s "weka.attributeSelection.Ranker -T -1.7976931348623157e+308 -N -1" -i "data/datasets/eye-movements.arff" -x 5 -n 123
         > "fs_outputs/heart-c_FS0.txt"

     """

    def __init__(self, config: FeatureSelectionStepSchema):
        self.config = config

    def generate_fs_weka_commands(self, input_file_path: str, mapping_csv_file) -> [FSCommandWithInfo]:
        """ this takes options for weka evaluation class and weka search method class and calculates all the neccessary
        stuff for building the java command for executing attribute selection weka.filters.supervised.attribute.AttributeSelection
        We use double quotes in the commands for WEKA
          """

        with open(input_file_path) as arff_file:
            dataframe_arff = DataFrameArff(arff_data=arff.load(arff_file))

        # TODO: label might not be the last one. Last column might contain OD score, check if following solution works

        index_of_class_attribute = len(dataframe_arff.arff_data()["attributes"]) - 1
        if dataframe_arff.arff_data()["attributes"][index_of_class_attribute] == OD_VALUE_NAME:
            print("OD_VALUE_COLUMN recognized")
            index_of_class_attribute -= 1

        _run_args = []
        counter = 0
        for feature_selection_config in self.config.selection_methods:
            # the command begins with "java", "-Xmx1024m" max heap size and "-cp" classpath specification
            _run_args += ["java", "-Xmx1024m", "-cp", self.config.weka_jar_path, "weka.filters.supervised.attribute.AttributeSelection"]
            # add input file path
            _run_args += ["-i", input_file_path]
            # specify index of the label class (the last one)
            _run_args += ["-c", str(index_of_class_attribute+1)]  # in weka, arff columns are indexed from one
            # specify search method and their arguments
            _run_args += ["-S", get_weka_command_from_config(feature_selection_config.search_class)]
            # specify evaluation mthod
            _run_args += ["-E", get_weka_command_from_config(feature_selection_config.eval_class)]
            # this hash is here to uniquely identify output files. It prevents new files with different settings
            # from overwriting older files with different settings
            hash_md5 = hashlib.md5(feature_selection_config.json().encode()).hexdigest()
            _output_file_path = _assert_trailing_slash(self.config.output_folder_path) + \
                                ".".join(os.path.basename(input_file_path).split(".")[:-1]) + "_FS-" + hash_md5 + ".txt"
            # TODO: add redirection of output where we call the subprocess
            _run_args += ["-o", _output_file_path]

            # this is the easiest way how to store FS configuration and file locations
            mapping_csv_file.write(
                input_file_path + "," + _output_file_path + "," + feature_selection_config.json().replace(",", ";"))

            file_split = input_file_path.split("_")
            # TODO: remove after testing is dono. Training data always have to be in folds, but here we fix if they are not
            if len(file_split) < 2:
                file_split.append("0")
            yield FSCommandWithInfo(args=_run_args,
                                    ds=file_split[0],
                                    fold=file_split[1],
                                    eval=feature_selection_config.eval_class.class_name,
                                    out=_output_file_path)
