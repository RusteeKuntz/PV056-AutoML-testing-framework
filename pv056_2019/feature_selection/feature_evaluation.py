import argparse
import re
import os
from pv056_2019.schemas import WekaClassCommandSchema, FeatureSelectionStepSchema, FeatureSelectionConfigurationSchema


def _nest_quotes(string, which_quotes="\""):
    # TODO: escaping escape slashes themselves might not be necessary, check it later
    # string = re.sub(r"\\", r"\\\\", string)
    return re.sub(which_quotes, "\\" + which_quotes, string)


def _nest_double_quotes(string):
    return _nest_quotes(string, "\"")


def _assert_trailing_slash(string):
    if string[-1] != "/":
        return string + "/"


def get_weka_command_from_config(config: WekaClassCommandSchema) -> str:
    """
    This will return a command containing an executable WEKA class with arguments, enclosed in double quotes,
    ready to be placed as a parameter to another executable WEKA class.
    """
    _command = "\"" + config.class_name
    for param_name, param_val in config.parameters:
        _command += " -" + param_name + " " + _nest_double_quotes(param_val)

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



    def get_command(self, input_file_path: str, feature_selection_config: FeatureSelectionConfigurationSchema) -> str:
        """ this takes options for weka evaluation class and weka search method class and calculates all the neccessary
        stuff for building the java command
        We use double quotes in the commands for WEKA
          """

        # the command begins with "java" and "-cp" classpath specification
        _command = "java -cp " + self.config.weka_jar_path + " weka.filters.supervised.attribute.AttributeSelection"
        # add input file path
        _command += " -i \"" + input_file_path + "\""
        # specify index of the label class (the last one)
        _command += " -c " + index_of_class_attribute
        # specify number of folds used in cross-validation
        _command += " -x " + number_of_folds
        # set seed for picking folds in cross-validation
        _command += " -n " + cross_validation_seed
        # specify evaluation/search method and their arguments
        _command += " -E " + get_weka_command_from_config(feature_selection_config.eval_class)
        _command += " -S " + get_weka_command_from_config(feature_selection_config.search_class)

        # add redirection to a file

        _output_file_path = _assert_trailing_slash(self.config.output_folder) + \
            os.path.basename(input_file_path).split(".")[:-1] + "_FR.txt"
        _command += " > \"" + _output_file_path + "\""

        return _command
