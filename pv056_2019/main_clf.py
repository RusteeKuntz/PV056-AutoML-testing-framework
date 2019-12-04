import argparse
import csv
import json
import os
import time
import subprocess
import sys

from multiprocessing import Process, Queue, Manager

from pv056_2019.classifiers import ClassifierManager
from pv056_2019.schemas import RunClassifiersCongfigSchema

times_file: str
timeout: int


def _valid_config_path(path):
    import argparse

    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("Invalid path to config file.")
    else:
        return path


def weka_worker(queue, blacklist):
    while not queue.empty():
        args = queue.get()
        time_diff: float

        file_split = args[6].split("/")[-1].split("_")
        dataset = file_split[0]
        clf = args[16].split(".")[-1]

        if not (clf, dataset) in blacklist:
            try:
                start_time = time.time()
                subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
                time_diff = time.time() - start_time
            except subprocess.TimeoutExpired:
                time_diff = timeout
                blacklist.append((clf, dataset))
        else:
            time_diff = timeout

        clf_hex = args[10].split("/")[-1].split("_")[-1].split(".")[0]
        fold = file_split[1]
        od_hex = file_split[2]
        rm = file_split[3].split("-")[1]

        with open(times_file, "a") as tf:
            print(",".join([dataset, fold, clf, clf_hex, od_hex, rm, str(time_diff)]), file=tf)

        print(";".join([args[16], args[6], args[8]]), flush=True)


def main():
    parser = argparse.ArgumentParser(description="PV056-AutoML-testing-framework")
    parser.add_argument(
        "-c",
        "--config-clf",
        type=_valid_config_path,
        help="path to classifiers config file",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--datasets-csv",
        type=_valid_config_path,
        help="Path to csv with data files",
        required=True,
    )
    args = parser.parse_args()

    with open(args.config_clf, "r") as config_file:
        conf = RunClassifiersCongfigSchema(**json.load(config_file))

    global times_file
    global timeout
    times_file = conf.times_output
    timeout = conf.timeout
    with open(conf.times_output, "w") as tf:
        print("dataset,fold,clf,clf_hex,od_hex,removed,clf_time", file=tf)
    # open(BLACKLIST_FILE, "w+").close()

    with open(args.datasets_csv, "r") as datasets_csv_file:
        reader = csv.reader(datasets_csv_file, delimiter=",")
        datasets = sorted([row for row in reader], key=lambda x: os.path.getsize(x[0]))

    clf_man = ClassifierManager(conf.output_folder, conf.weka_jar_path)

    with Manager() as manager:
        blacklist = manager.list()

        queue = Queue()
        clf_man.fill_queue_and_create_configs(queue, conf.classifiers, datasets)

        pool = [Process(target=weka_worker, args=(queue, blacklist)) for _ in range(conf.n_jobs)]

        try:
            [process.start() for process in pool]
            [process.join() for process in pool]
        except KeyboardInterrupt:
            [process.terminate() for process in pool]
            print("\nInterupted!", flush=True, file=sys.stderr)

    print("Done")


if __name__ == "__main__":
    main()
