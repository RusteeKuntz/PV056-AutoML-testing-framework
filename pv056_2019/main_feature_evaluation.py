import argparse
import json
# TODO: Move the original definition into Utils (if possible)
from pv056_2019.main_clf import _valid_config_path
from pv056_2019.schemas import FeatureSelectionStepSchema
from pv056_2019.feature_selection.feature_evaluation import FeatureSelectionManager, _assert_trailing_slash


def main():
    parser = argparse.ArgumentParser(description="PV056-AutoML-testing-framework")
    parser.add_argument(
        "-c",
        "--config-fs",
        type=_valid_config_path,
        help="path to feature selection config file",
        required=True,
    )
    parser.add_argument(
        "-fs",
        "--fs-mapping-csv",
        help="path to a csv file which contains datasets used for FS and their respective result files",
        required=False,
    )
    # TODO: uncomment later when implementing real data manipulation
    # parser.add_argument(
    #     "-d",
    #     "--datasets-csv",
    #     type=_valid_config_path,
    #     help="Path to csv file that contains data files mappings and locations",
    #     required=True,
    # )
    args = parser.parse_args()

    # load config file
    with open(args.config_fs, "r") as config_file:
        conf = FeatureSelectionStepSchema(**json.load(config_file))
    print(conf)

    # TODO: Here implement getting filepaths from datasets.csv instead of temporary hardcoded list of files
    dataset_paths = ["A:\Adam\Škola\BAKPR\data\datasets\eye-movements.arff",
             "A:\Adam\Škola\BAKPR\data\datasets\glass.arff",
             "A:\Adam\Škola\BAKPR\data\datasets\heart-c.arff",
             "A:\Adam\Škola\BAKPR\data\datasets\hepatitis.arff"
             ]

    fs_manager = FeatureSelectionManager(conf)
    # default path to a CSV with datasets and their FS results is with the results
    fs_mapping_csv_path = _assert_trailing_slash(fs_manager.config.output_folder_path) + "fs_mapping.csv" if args.fs_mapping_csv is None else args.fs_mapping_csv

    WEKA_commands = []
    with open(fs_mapping_csv_path, "w") as fs_mapping_csv:
        # Here we open a file to record which file was used for feature selection, where the result was stored and which
        # parameters were used in the process (config). The file is then passed to the method below to write the actual
        # dataset_path records, because it is easier.
        # Format is: input-file, output-file, config-json
        fs_mapping_csv.write("input_file, output_file, config\n")
        for dataset_path in dataset_paths:
            for command in fs_manager.get_commands(dataset_path, fs_mapping_csv):
                WEKA_commands.append(command)
                print(command)




if __name__ == '__main__':
    main()