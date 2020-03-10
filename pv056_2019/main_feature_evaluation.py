import argparse
import json
# TODO: Move the original definition into Utils (if possible)
from pv056_2019.main_clf import _valid_config_path
from pv056_2019.schemas import FeatureSelectionStepSchema


def main():
    parser = argparse.ArgumentParser(description="PV056-AutoML-testing-framework")
    parser.add_argument(
        "-c",
        "--config-fs",
        type=_valid_config_path,
        help="path to feature selection config file",
        required=True,
    )
    # TODO: uncomment later when implementind real data manipulation
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

    # TODO: Here implement getting filepaths from datasets.csv instead of temporary hardcoded list of files
    files = ["A:\Adam\Škola\BAKPR\data\datasets\eye-movements.arff",
             "A:\Adam\Škola\BAKPR\data\datasets\glass.arff",
             "A:\Adam\Škola\BAKPR\data\datasets\heart-c.arff",
             "A:\Adam\Škola\BAKPR\data\datasets\hepatitis.arff"
             ]


if __name__ == '__main__':
    main()