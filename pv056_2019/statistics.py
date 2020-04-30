import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from pv056_2019.schemas import StatisticsSchema


def compile_reg(s):
    try:
        return re.compile(s)
    except Exception as e:
        print("Regex error:", e, file=sys.stderr)
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Script for counting basic statistic (Accuracy, )"
    )
    parser.add_argument("--config-file", "-c", required=True, help="JSON configuration")
    parser.add_argument("--datasets-csv-in", "-di", required=True,
                        help="CSV file with paths to predictions, test files and configurations histories")
    parser.add_argument("--datasets-csv-baseline", "-db", required=False,
                        help="CSV file with paths to predictions, test files and configurations histories")

    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        conf: StatisticsSchema = StatisticsSchema(**json.load(json_file))

    #reg = re.compile(r"removed-0*")
    # Previously these files were listed from a directory
    #files = sorted([x for x in os.listdir(conf.results_dir) if x.endswith(".csv")])

    with open(args["datasets_csv_in"], "r") as csv_in:
        #csv_reader = csv.reader(csv_in, delimiter=",")
        csv_rows = sorted([row for row in csv.reader(csv_in, delimiter=",")], key=lambda x: os.path.getsize(x[0]))

    #pattern = compile_reg(conf.pattern)

    # TODO: migh be used again in future versions
    config_file_paths = [
        # Previously these jsons were also listed from a directory
        # x for x in os.listdir(conf.results_dir) if x.endswith(".json")
    ]

    config_dict = {}
    # for config_file_path in config_file_paths:
    #     with open(os.path.join(conf.results_dir, config_file_path)) as config_file:
    #         basename = os.path.basename(config_file_path)
    #         conf_hash = basename.split("_")[1].replace(".json", "")
    #         config_dict[conf_hash] = json.load(config_file)

    data = []
    # We actually take into account future upgrades, which might include non-identical number of preprocessing steps applied
    greatest_steps_count = 0
    for csv_row in csv_rows:
        #if not pattern.match(fl):
        #    continue
        prediction_file = csv_row[0]
        conf_path = csv_row[2]


        file_split = prediction_file.split("_")
        #file_split[-1] = file_split[-1].replace(".csv", "")

        # We dont care about filenames anymore
        # if "removed-" in file_split[-1]:
        #     file_split[-1] = reg.sub("", file_split[-1])
        # else:
        #     file_split.append(0)

        #datest, split, classifier, conf_hash, removed = file_split

        # we still do care about json filenames (practical reasons) Here we extract a hash fro that filename
        conf_hash = conf_path.split(".")[0]
        if conf_hash not in config_dict:
            with open(conf_path) as config_file:
                config_dict[conf_hash] = json.load(config_file)

        conf = config_dict[conf_hash]

        classifier = conf["model_config"].get("class_name").split(".")[-1]
        classifier_args = conf["model_config"].get("args")
        steps_count = conf["model_config"].get("steps_count")
        if steps_count > greatest_steps_count:
            greatest_steps_count = steps_count

        #od_name = config_dict[conf_hash]["od_configs"].get("name", "")

        dataframe = pd.read_csv(prediction_file)
        all_results = dataframe.shape[0]
        accuracy = np.sum(dataframe["error"] != "+") / all_results

        #data.append([datest, split, classifier, od_name, removed, conf_hash, accuracy])
        data.append([])

    #od_times = pd.read_csv(conf.od_times_path)
    #clf_times = pd.read_csv(conf.clf_times_path)

    #times = od_times.merge(clf_times, on=["dataset", "fold", "od_hex"], how="outer")
    #times['fold'] = times['fold'].astype(str)
    #times.loc[(times.removed == 0), 'od_time'] = 0.0
    #times['total_time'] = times['od_time'] + times['clf_time']


    headers = [
        "dataset",
        "fold",
        "clf",
        "clf_hex",
        *["step_{}".format(x) for x in range(greatest_steps_count)],
        "accuracy",
    ]


    dataframe = pd.DataFrame(data, columns=headers)
    dataframe['removed'] = dataframe['removed'].astype(float)

    dataframe = dataframe.merge(times, on=["dataset", "fold", "clf", "clf_hex", "removed"], how="right")

    dataframe['clf_params'] = [re.sub("'|\"|,", "", str(config_dict[ax]["model_config"].get("args")))
                               for ax in dataframe['clf_hex']]
    dataframe['od_params'] = [str(config_dict[ax]["ad_config"].get("parameters")).replace(",", ";").replace("'", "")
                              for ax in dataframe['clf_hex']]

    if conf.aggregate:
        dataframe = dataframe.groupby(
            ["dataset", "clf", "clf_family", "clf_params", "od_name", "od_params", "removed"]
        ).mean()
        dataframe = dataframe.loc[:, dataframe.columns != "fold"]

    dataframe = dataframe.round(5)
    print(dataframe.to_csv())
    with open(conf.output_table, "w+") as ot:
        print(dataframe.to_csv(), file=ot)
    backup_ts = "backups/" + conf.output_table.split("/")[-1].replace(".csv", datetime.now()
                                                                      .strftime("_backup_%d-%m-%Y_%H-%M.csv"))
    with open(backup_ts, "w+") as ot:
        print(dataframe.to_csv(), file=ot)


if __name__ == "__main__":
    main()
