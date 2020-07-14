import argparse

import pandas as pd

from pv056_2019.schemas import GraphStepSchema
from pv056_2019.utils import valid_path


def sort_boxplots_on_mean(e: [pd.Series, str]):
    return -e[0].mean()

def sort_boxplots_on_mean_inverse(e: [pd.Series, str]):
    return e[0].mean()

def sort_boxplots_on_label(e: [pd.Series, str]):
    return e[1]

def sort_boxplots_on_median(e: [pd.Series, str]):
    return e[0].median()

SORT_FUNCTIONS = {"label": sort_boxplots_on_label,
                  "mean": sort_boxplots_on_mean,
                  "inv_mean": sort_boxplots_on_mean_inverse,
                  "median": sort_boxplots_on_median}

def setup_arguments()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PV056-AutoML-testing-framework")
    parser.add_argument(
        "-c",
        "--config-graph",
        type=lambda x: valid_path(x, "Invalid path to configuration json file!"),
        help="path to visualisation config file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-png",
        help="path to a png file which contains visualized results in a nice graph",
        required=False,
    )
    parser.add_argument(
        "-di",
        "--datasets-csv-in",
        type=lambda x: valid_path(x, "Invalid path to input csv file!"),
        help="Path to csv file that contains results from a workflow",
        required=True,
    )
    return parser

def prepare_data(args, conf: GraphStepSchema) -> (pd.DataFrame, str):
    print("Reading results")
    df = pd.read_csv(args.datasets_csv_in)
    print("Sorting dataset.")
    if conf.sort_by_column is not None:
        sorted_df = df.sort_values(by=conf.sort_by_column)

    return df, args.output_png
