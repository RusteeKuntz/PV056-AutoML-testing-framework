import argparse
import csv
from datetime import datetime
import json
import os
from multiprocessing.managers import SyncManager

import resource
import subprocess
import sys
from multiprocessing import Process, Queue, Manager

from pv056_2019.classifiers import ClassifierManager, CLFCommandWithInfo
from pv056_2019.schemas import RunClassifiersCongfigSchema

#blacklist_file: str
#times_file: str
#timeout: int


def _valid_config_path(path):
    import argparse

    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("Invalid path to config file.")
    else:
        return path


def weka_worker(queue,
                blacklist,
                timeout,
                times_file,
                backup_ts,
                counter
):
    while not queue.empty():
        this_counter = counter.get()
        counter.set(counter.get() + 1)
        print("Popping command " + str(this_counter) + " from queue")
        command_with_info: CLFCommandWithInfo = queue.get()
        args = command_with_info.args
        time_diff: float

        #file_split = args[6].split("/")[-1].split("_")
        dataset = command_with_info.dataset  # file_split[0]
        clf = command_with_info.clf_classname  # args[16].split(".")[-1]
        print("BLACKLIST: ", blacklist, "\nTUPLE: ", (clf, dataset), (clf, dataset) in blacklist)
        if not (clf, dataset) in blacklist:
            try:
                time_start = resource.getrusage(resource.RUSAGE_CHILDREN)[0]
                result = subprocess.run(args,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        timeout=timeout)
                # these are mostly debugging outputs. Turn them on if you want to know what goes wrong.
                #print("RESULT:", clf, dataset,
                      #"\nERR:", result.stderr,
                      #"\nOUT:", result.stdout,
                      #"\nRC:", result.returncode)
                # print(" ".join(args))
                # print(result.stdout)
                # print(result.stderr)
                time_end = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

                time_diff = time_end - time_start
            except subprocess.TimeoutExpired:
                time_diff = timeout
                blacklist.append((clf, dataset))
                print(clf, dataset, "TIMED OUT! Appending combination to a temporary blacklist. Continuing.")
        else:
            time_diff = timeout

        clf_fam = ".".join(command_with_info.clf_classname.split(".")[2:-1])
        fold = command_with_info.fold  # file_split[1]

        # TODO: this has to be replaced by list of settings successively applied during workflow
        clf_hex = ""  # args[10].split("/")[-1].split("_")[-2]
        od_hex = ""  # file_split[2]
        rm = ""  # file_split[3].split("-")[1]

        row_string = ",".join([dataset, fold, clf, clf_fam, clf_hex, command_with_info.settings.replace(",", ";"), str(time_diff)])
        with open(times_file, "a") as tf:
            print(row_string, file=tf)
        with open(backup_ts, "a") as tf:
            print(row_string, file=tf)

        # args[16] is actually equal to full clf classname
        # args[6] is actually a path to a train file
        print(str(this_counter) + ". job done.", flush=True)



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
        "-di",
        "--datasets-csv-in",
        type=_valid_config_path,
        help="Path to csv with train/test/configs filepaths.",
        required=True,
    )
    parser.add_argument(
        "-do",
        "--datasets-csv-out",
        help="Path to csv with predictions, test files, configs filepaths.",
        required=True,
    )
    args = parser.parse_args()
    print("starting CLF step")

    with open(args.config_clf, "r") as config_file:
        conf = RunClassifiersCongfigSchema(**json.load(config_file))

    # here we read the dataset.csv with tuples of train/test files
    with open(args.datasets_csv_in, "r") as datasets_csv_file:
        reader = csv.reader(datasets_csv_file, delimiter=",")
        # here we get an array of datasets.csv lines arranged by the size of the file in first column of each row
        # paths are checked before usage. If they do not exist, 0 is used as size for comparison
        unsorted_dataset_rows = [row for row in reader]
        # for x in unsorted_dataset_rows:
        #     print(x)
        #     if os.path.exists(x[0]):
        #         print(x[0] + " exists")
        #     else:
        #         print(x[0] + " DOES NOT exists")
        datasets = sorted(unsorted_dataset_rows,
                          key=lambda x: os.path.getsize(x[0]) if os.path.exists(x[0]) else 0)

    # They taught us not to use GLOBAL variables, so I replaced them with normal variables and pass them as params
    #global times_file
    #global blacklist_file
    #global timeout
    #times_file = conf.times_output
    #blacklist_file = conf.blacklist_file
    #timeout = conf.timeout
    if len(datasets) == 0:
        print("NO datasets available, returning 0.")
        return 0

    # count the number of path to config jsons (the number of steps applied)
    number_of_steps = len(datasets[0][2:])
    headdings = "dataset,fold,clf,clf_family,clf_time," + ",".join(["step_{}".format(x) for x in range(number_of_steps)])
    with open(conf.times_output, "w+") as tf:
        print(headdings, file=tf, flush=True)
    backup_ts = "backups/" + conf.times_output.split("/")[-1].replace(".csv", datetime.now()
                                                                     .strftime("_backup_%d-%m-%Y_%H-%M.csv"))
    with open(backup_ts, "w+") as tf:
        print(headdings, file=tf)

    #open(blacklist_file, "a+").close()
    open(conf.times_output, "w").close()


    clf_man = ClassifierManager(conf.output_folder, conf.weka_jar_path)

    with Manager() as manager:
        # create concurrency safe list for sharing between threads/processes
        blacklist = manager.list()
        counter = manager.Value('i', 0)
        with open(conf.blacklist_file, "r") as bf:
            # read files from blacklist file and put them into the actual list
            for i in bf:
                blacklist.append(tuple(i.replace("\n", "").split(',')))

        queue = Queue()
        print("FILLING THE QUEUE")
        clf_man.fill_queue_and_create_configs(queue, conf.classifiers, datasets, args.datasets_csv_out)
        print("BACK IN MAIN")

        # TODO xbajger: previously, here as an arg to weka_worker a "backup_ts" file was supplied. It shan't be needed
        pool = [Process(target=weka_worker, args=(queue, blacklist, conf.timeout, conf.times_output, backup_ts, counter)) for _ in range(conf.n_jobs)]

        try:
            [process.start() for process in pool]
            [process.join() for process in pool]
        except KeyboardInterrupt:
            [process.terminate() for process in pool]
            print("\nInterupted!", flush=True, file=sys.stderr)

    print("Done")


if __name__ == "__main__":
    main()
