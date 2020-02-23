import argparse
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

    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        conf = StatisticsSchema(**json.load(json_file))

    reg = re.compile(r"removed-0*")
    files = sorted([x for x in os.listdir(conf.results_dir) if x.endswith(".csv")])

    pattern = compile_reg(conf.pattern)

    config_file_paths = [
        x for x in os.listdir(conf.results_dir) if x.endswith(".json")
    ]

    config_dict = {}
    for config_file_path in config_file_paths:
        with open(os.path.join(conf.results_dir, config_file_path)) as config_file:
            basename = os.path.basename(config_file_path)
            conf_hash = basename.split("_")[1].replace(".json", "")
            config_dict[conf_hash] = json.load(config_file)

    headers = [
        "dataset",
        "fold",
        "clf",
        "od_name",
        "removed",
        "clf_hex",
        "accuracy",
    ]
    data = []
    for fl in files:
        if not pattern.match(fl):
            continue

        file_split = fl.split("_")
        file_split[-1] = file_split[-1].replace(".csv", "")

        if "removed-" in file_split[-1]:
            file_split[-1] = reg.sub("", file_split[-1])
        else:
            file_split.append(0)

        datest, split, classifier, conf_hash, removed = file_split

        classifier = config_dict[conf_hash]["model_config"].get("class_name").split(".")[-1]

        od_name = config_dict[conf_hash]["ad_config"].get("name", "")

        dataframe = pd.read_csv(os.path.join(conf.results_dir, fl))
        all_results = dataframe.shape[0]
        accuracy = np.sum(dataframe["error"] != "+") / all_results

        data.append([datest, split, classifier, od_name, removed, conf_hash, accuracy])

    od_times = pd.read_csv(conf.od_times_path)
    clf_times = pd.read_csv(conf.clf_times_path)

    times = od_times.merge(clf_times, on=["dataset", "fold", "od_hex"], how="outer")
    times['fold'] = times['fold'].astype(str)
    times.loc[(times.removed == 0), 'od_time'] = 0.0
    times['total_time'] = times['od_time'] + times['clf_time']

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
