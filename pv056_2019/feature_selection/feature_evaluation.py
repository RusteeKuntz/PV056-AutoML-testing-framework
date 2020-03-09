import argparse
import re
import os
from pv056_2019.schemas import FeatureSelectionSchema, FeatureSelectionStepSchema


def _nest_quotes(string, which_quotes="\""):
    # TODO: escaping escape slashes themselves might not be necessary
    # string = re.sub(r"\\", r"\\\\", string)
    return re.sub(which_quotes, "\\" + which_quotes, string)


def _nest_double_quotes(string):
    return _nest_quotes(string, "\"")


def _assert_trailing_slash(string):
    if string[-1] != "/":
        return string + "/"


class FeatureSelectionManager():
    """ This class serves as a tool for building full java command for command line to execute in an external process
    example commandline input:
    java -cp A:\Windows\Programy\Weka-3-8-4\weka.jar
        weka.attributeSelection.InfoGainAttributeEval -i input_file_path
        -s "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1" > output_file_path
     """

    def __init__(self, config: FeatureSelectionStepSchema):
        self.config = config

    def get_command(self, input_file_path: str, feature_selection_config: FeatureSelectionSchema) -> str:
        """ this takes options for weka evaluation class and weka search method class and calculates all the neccessary
        stuff for building the java command
        We use double quotes in the commands for WEKA
          """

        # the command begins with "java" and "-cp" classpath specification
        _command = "java -cp " + self.config.weka_jar_path

        # specify evaluation method and its arguments
        _command += " " + feature_selection_config.eval_method.name
        # add input file path
        _command += " -i \"" + input_file_path + "\""
        for key, arg in feature_selection_config.search_method.parameters:
            _command += " -" + key + " " + arg

        # build search method configuration
        search_command = feature_selection_config.search_method.name + ' '
        for key, arg in feature_selection_config.search_method.parameters:
            # re.escape
            search_command += " -" + key + " " + _nest_double_quotes(arg)

        # add search method to the command as a parameter for evaluation class
        _command += " -s \"" + _nest_double_quotes(search_command) + "\""

        # add redirection to a file

        _output_file_path = _assert_trailing_slash(self.config.output_folder) + \
            os.path.basename(input_file_path).split(".")[:-1] + "_FR.txt"
        _command += " > \"" + _output_file_path + "\""

        return _command
