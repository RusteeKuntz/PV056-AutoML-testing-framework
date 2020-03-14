import argparse
import re, os, json

import arff

from pv056_2019.schemas import WekaClassCommandSchema, FeatureSelectionStepSchema, FeatureSelectionConfigurationSchema
from pv056_2019.data_loader import DataLoader, DataFrameArff
from pv056_2019.utils import OD_VALUE_NAME


def _nest_quotes(string, which_quotes="\""):
    # TODO: escaping escape slashes themselves might not be necessary, check it later
    # string = re.sub(r"\\", r"\\\\", string)
    print(which_quotes)
    print(string)
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
    _command = "\"" + config.class_name
    print("DEBUG: values of config.parameters")
    print(config.parameters)
    for param_name, param_val in config.parameters.items():
        _command += " -" + param_name + " " + _nest_double_quotes(str(param_val))

    return _command + "\""


class FeatureSelectionManager:
    """ This class serves as a tool for building full java command for command line to execute in an external process
    example commandline input:

    java -cp A:\Windows\Programy\Weka-3-8-4\weka.jar weka.filters.supervised.attribute.AttributeSelection
        -E "weka.attributeSelection.InfoGainAttributeEval "
        -S "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1"
        -c index_of_class_attribute
        -x number-of-folds-for-cross-validation
        -n random-seed-number
        > output_file_path
     """

    def __init__(self, config: FeatureSelectionStepSchema):
        self.config = config



    def get_commands(self, input_file_path: str, mapping_csv_file) -> [str]:
        """ this takes options for weka evaluation class and weka search method class and calculates all the neccessary
        stuff for building the java command
        We use double quotes in the commands for WEKA
          """

        with open(input_file_path) as arff_file:
            dataframe_arff = DataFrameArff(arff_data=arff.load(arff_file))

        # TODO: label might not be the last one. Last column might contain OD score, check if following solution works

        index_of_class_attribute = len(dataframe_arff.arff_data()["attributes"]) - 1
        if dataframe_arff.arff_data()["attributes"][index_of_class_attribute] == OD_VALUE_NAME:
            index_of_class_attribute -= 1

        counter = 0
        for feature_selection_config in self.config.selection_methods:
            # the command begins with "java" and "-cp" classpath specification
            _command = "java -cp " + self.config.weka_jar_path + " weka.filters.supervised.attribute.AttributeSelection"
            # add input file path
            _command += " -i \"" + input_file_path + "\""
            # specify index of the label class (the last one)
            _command += " -c " + str(index_of_class_attribute)
            # specify number of folds used in cross-validation
            _command += " -x " + str(feature_selection_config.n_folds)
            # set seed for picking folds in cross-validation
            _command += " -n " + str(feature_selection_config.cv_seed)
            # specify evaluation/search method and their arguments
            _command += " -E " + get_weka_command_from_config(feature_selection_config.eval_class)
            _command += " -S " + get_weka_command_from_config(feature_selection_config.search_class)

            # add redirection to a file
            _output_file_path = _assert_trailing_slash(self.config.output_folder_path) + \
                ".".join(os.path.basename(input_file_path).split(".")[:-1]) + "_FS" + str(counter) + ".txt"

            _command += " > \"" + _output_file_path + "\""

            # this is the easiest way how to store FS configuration and file locations
            mapping_csv_file.write(input_file_path + "," + _output_file_path + "," + json.dumps(feature_selection_config).replace(",", ";"))

            yield _command
