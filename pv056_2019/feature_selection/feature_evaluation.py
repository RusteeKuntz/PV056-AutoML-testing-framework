import argparse
import hashlib
import re, os, json

import arff

from pv056_2019.schemas import CommandSchema, FeatureSelectionStepSchema, \
    WekaFSFilterConfigurationSchema, CustomFSSchema
from pv056_2019.data_loader import DataLoader, DataFrameArff
from pv056_2019.utils import OD_VALUE_NAME, ID_NAME, CUSTOM, WEKA


class FSCommandWithInfo:
    is_custom: bool
    args: []
    input_path: str
    fold: str
    eval_method_name: str
    output_file_path: str

    def __init__(self,
                 is_custom,
                 args: [str],
                 ds: str,
                 # fold: str,
                 ev: str,
                 out: str) -> None:
        self.is_custom = is_custom
        self.args = args
        self.input_path = ds
        # self.fold = fold
        self.eval_method_name = ev
        self.output_file_path = out

    def __str__(self) -> str:
        return " ".join(self.args)


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


def get_weka_command_from_config(config: CommandSchema) -> str:
    """
    This will return a command containing an executable WEKA class with arguments, enclosed in double quotes
    ready to be placed as a parameter to another executable WEKA class.
    """
    _command = config.name
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
    """

    def __init__(self, config: FeatureSelectionStepSchema):
        self.config = config

    def generate_fs_weka_commands(self, datasets_mapping_csv, mapping_csv_file) -> [FSCommandWithInfo]:
        """ this takes a csv file with datasets and their configurations use by other workflow steps (if any)
        Such csv file contains 3 columns: train_dataset,test_dataset,configuration_json_path
        This method uses options for weka evaluation class and weka search method class and calculates all the necessary
        stuff for building the java command for executing attribute selection
        weka.filters.supervised.attribute.AttributeSelection on datasets from datasets_mapping_csv
        """
        # Format is: train-file, test-file, 1st config-json, 2nd config-json, ...
        # mapping_csv_file.write("train, test, config\n") #DO NOT WRITE HEADINGS into this csv

        # precompute hashes for configurations and json files to speed up execution and avoid redundancy
        fs_settings = []
        for feature_selection_config in self.config.selection_methods:
            # this hash is here to uniquely identify output files. It prevents new files with different settings
            # from overwriting older files with different settings
            hash_md5 = hashlib.md5(feature_selection_config.json().encode()).hexdigest()
            fs_config_json_basename = hash_md5 + ".json"

            with open(os.path.join(self.config.output_folder_path, fs_config_json_basename), "w") as config_json:
                config_json.write(feature_selection_config.json())

            fs_settings.append((feature_selection_config, hash_md5, fs_config_json_basename))

        # TODO: remove limitation to 20 datasets later
        limit_counter = 0
        for line in datasets_mapping_csv:
            if limit_counter > 20:
                break
            else:
                limit_counter += 1
            # split datasets csv line by commas, strip trailing EOL
            line_split = line.strip().split(",")
            # first two elements on any line contain train and test split paths.
            train_path = line_split[0]
            test_path = line_split[1]
            # next elements are paths to configuration jsons for steps executed before in order they were executed
            if len(line_split) > 2:
                conf_paths = line_split[2:]  # paths to a configuration jsons of previous steps
            else:
                conf_paths = []

            # pre-compute output directory name (assert trailing slash) and extract base name of the dataset
            _output_directory = _assert_trailing_slash(self.config.output_folder_path)
            _base_name = os.path.basename(train_path)

            for feature_selection_config, hash_md5, fs_config_json_basename in fs_settings:
                _fs_identifier = '_FS-' + hash_md5
                # this part checks if train file contains _train substring and places FS identifier string before of it
                if '_train' in _base_name:
                    _output_file_path = _output_directory + _base_name.replace('_train', _fs_identifier + '_train')
                else:
                    dot_split = _base_name.split('.')
                    _output_file_path = _output_directory + '.'.join(dot_split[:-1]) + _fs_identifier + '.' + dot_split[
                        -1]

                # here we write mapping of train and test files along with a history of pre-processing configurations
                mapping_csv_file.write(
                    ",".join(
                        [_output_file_path, test_path] +
                        conf_paths +
                        [_output_directory + fs_config_json_basename]
                    ) + "\n"
                )
                mapping_csv_file.flush()

                # generate command with info appropriately for used library
                if feature_selection_config.source_library == WEKA:
                    # here we prepare filters for currently useless columns that should not be considered for FS
                    filters = ['-F', 'weka.filters.unsupervised.attribute.RemoveByName -E ^{}$'.format(  # noqa
                        ID_NAME
                    )] + ['-F', 'weka.filters.unsupervised.attribute.RemoveByName -E ^{}$'.format(
                        OD_VALUE_NAME
                    )]

                    fs_filter_args = ""
                    # currently: don't specify index of the label class (default is the last one).

                    # specify search method and its arguments
                    fs_filter_args += ' -S ' + '"' + _nest_double_quotes(
                        get_weka_command_from_config(feature_selection_config.search_class)) + '"'
                    # specify evaluation method and its arguments
                    fs_filter_args += ' -E ' + '"' + _nest_double_quotes(get_weka_command_from_config(
                        feature_selection_config.eval_class)) + '"'

                    filters += ['-F', 'weka.filters.supervised.attribute.AttributeSelection {}'.format(
                        fs_filter_args
                    )]

                    # the command begins with 'java', '-Xmx1024m' max heap size and '-cp' classpath specification
                    _run_args = ['java', '-Xmx1024m', '-cp', self.config.weka_jar_path,
                                 # add input/output file path and
                                 # filters
                                 'weka.filters.MultiFilter'] + filters + ['-i', train_path, '-o', _output_file_path]

                    # TODO: remove after testing is done.
                    #  Training data always have to be in folds, but here we fix if they are not
                    # if len(file_split) < 2:
                    #    file_split.append("0")

                    # print(_run_args)

                    yield FSCommandWithInfo(is_custom=True,
                                            args=_run_args,
                                            ds=train_path,
                                            # fold=file_split[1],
                                            ev=feature_selection_config.eval_class.class_name,
                                            out=_output_file_path)
                elif feature_selection_config.source_library == CUSTOM:
                    # here we load config into pydantic Schema to apply validation before running teh experiment
                    conf: CustomFSSchema = CustomFSSchema(**feature_selection_config.dict())
                    yield FSCommandWithInfo(is_custom=False,
                                            args=conf.parameters,
                                            ds=train_path,
                                            ev=conf.name,
                                            out=_output_file_path)
