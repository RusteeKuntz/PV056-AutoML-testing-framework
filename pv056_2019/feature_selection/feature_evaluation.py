import hashlib
import os, json

from pv056_2019.data_loader import DataLoader
from pv056_2019.schemas import CommandSchema, FeatureSelectionStepSchema, \
    WekaFSFilterConfigurationSchema, CustomFSSchema, ScikitFSSchema, FSStepSchema
from pv056_2019.utils import OD_VALUE_NAME, ID_NAME, CUSTOM, WEKA, SCIKIT, _nest_double_quotes, _assert_trailing_slash


class FSJobWithInfo:
    is_cmd: bool
    args: [str] or ScikitFSSchema or CustomFSSchema
    input_path: str
    fold: str
    eval_method_name: str
    output_file_path: str
    mapping_csv_line: str

    def __init__(self,
                 is_cmd,
                 args: [str] or ScikitFSSchema or CustomFSSchema,
                 ds: str,
                 # fold: str,
                 ev: str,
                 out: str,
                 csv_line: str) -> None:
        self.is_cmd = is_cmd
        self.args = args
        self.input_path = ds
        # self.fold = fold
        self.eval_method_name = ev
        self.output_file_path = out
        self.mapping_csv_line = csv_line

    def __str__(self) -> str:
        return " ".join(self.args)


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

    def generate_fs_jobs(self, datasets_mapping_csv) -> [FSJobWithInfo]:
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
        for fs_conf_dict in self.config.selection_methods:
            # this hash is here to uniquely identify output files. It prevents new files with different settings
            # from overwriting older files with different settings
            conf_string = json.dumps(fs_conf_dict, sort_keys=True)
            hash_md5 = hashlib.md5(conf_string.encode(encoding="UTF-8")).hexdigest()
            fs_config_json_basename = hash_md5 + ".json"

            with open(os.path.join(self.config.output_folder_path, fs_config_json_basename), "w") as config_json:
                config_json.write(conf_string)

            fs_settings.append((fs_conf_dict, hash_md5, fs_config_json_basename))


        for line in datasets_mapping_csv:
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

            # pre-compute output directory name and extract base name of the dataset
            #_output_directory = _assert_trailing_slash()
            _base_name = os.path.basename(train_path)

            #print("FS SETTINGS:", fs_settings)
            for fs_conf_dict, hash_md5, fs_config_json_basename in fs_settings:

                fs_conf: FSStepSchema = FSStepSchema(**fs_conf_dict)
                _fs_identifier = '_FS' + hash_md5

                # TODO xbajger: Remove this "_train" gymnastic, it should be obsolete to keep that string in filenames
                # this part checks if train file contains _train substring and places FS identifier string before of it
                if '_train' in _base_name:
                    _output_file_path = os.path.join(
                        self.config.output_folder_path,
                        _base_name.replace('_train', _fs_identifier + '_train')
                    )
                else:
                    dot_split: [str] = _base_name.split('.')
                    _output_file_path = os.path.join(self.config.output_folder_path, '.'.join(dot_split[:-1]) + _fs_identifier + '.'
                                                     + dot_split[-1])

                # here we create mapping of train and test files along with a history of pre-processing configurations

                mapping_csv_file_line = ",".join(
                    [_output_file_path, test_path,
                     *conf_paths,
                     os.path.join(self.config.output_folder_path, fs_config_json_basename)]
                ) + "\n"

                # generate command with info appropriately for used library
                if fs_conf.source_library == WEKA:
                    wfs_conf: WekaFSFilterConfigurationSchema = WekaFSFilterConfigurationSchema(**fs_conf_dict)
                    if wfs_conf.search_class.name == "weka.attributeSelection.Ranker" and\
                        isinstance(wfs_conf.search_class.parameters["N"], str) and\
                        "%" in wfs_conf.search_class.parameters["N"]:
                            percent = float(wfs_conf.search_class.parameters["N"].replace("%", "").strip())
                            _in_df = DataLoader._load_arff_file(train_path)
                            f_count = _in_df.shape[1] - 1

                            # here we modify the original "-N" parameter for Ranker, so that it contains valid integer
                            n_param = round(percent*f_count)

                            #wfs_conf.search_class.parameters["N"] = n_param
                            sc: CommandSchema = CommandSchema(**{"name": wfs_conf.search_class.name, "parameters": {**wfs_conf.search_class.parameters}})
                            sc.parameters["N"] = n_param
                            wfs_conf.search_class = sc

                            print("Shape of dataset", train_path, f_count, n_param)
                            print("Replaced %")


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
                        get_weka_command_from_config(wfs_conf.search_class)) + '"'
                    # specify evaluation method and its arguments
                    fs_filter_args += ' -E ' + '"' + _nest_double_quotes(get_weka_command_from_config(
                        wfs_conf.eval_class)) + '"'

                    filters += ['-F', 'weka.filters.supervised.attribute.AttributeSelection {}'.format(
                        fs_filter_args
                    )]

                    # the command begins with 'java', '-Xmx1024m' max heap size and '-cp' classpath specification
                    _run_args = ['java', '-Xmx2048m', '-cp', self.config.weka_jar_path,
                                 # add input/output file path and
                                 # filters
                                 'weka.filters.MultiFilter'] + filters + ['-i', train_path, '-o', _output_file_path]

                    # "/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-2_e1f9d3e3d84ace3422e3d715688add12_OD-414bd82745e9a87176dd4401a880a9ff_RM5.00_FS-1e4787c934fd048ab9e8fdd605d78578_train.arff"

                    # TODO: remove after testing is done.
                    #  Training data always have to be in folds, but here we fix if they are not
                    # if len(file_split) < 2:
                    #    file_split.append("0")

                    print(_run_args)

                    yield FSJobWithInfo(is_cmd=True,
                                        args=_run_args,
                                        ds=train_path,
                                        ev=wfs_conf.eval_class.name,
                                        out=_output_file_path,
                                        csv_line=mapping_csv_file_line)
                elif fs_conf.source_library == CUSTOM:
                    # here we load config into pydantic Schema to apply validation before running the experiment
                    cfs_conf: CustomFSSchema = CustomFSSchema(**fs_conf_dict)
                    yield FSJobWithInfo(is_cmd=False,
                                        args=cfs_conf,
                                        ds=train_path,
                                        ev=cfs_conf.name,
                                        out=_output_file_path,
                                        csv_line=mapping_csv_file_line)
                elif fs_conf.source_library == SCIKIT:
                    # here we load config into pydantic Schema to apply validation before running the experiment
                    sfs_conf: ScikitFSSchema = ScikitFSSchema(**fs_conf_dict)
                    yield FSJobWithInfo(is_cmd=False,
                                        args=sfs_conf,
                                        ds=train_path,
                                        ev=sfs_conf.fs_method.name,
                                        out=_output_file_path,
                                        csv_line=mapping_csv_file_line)
