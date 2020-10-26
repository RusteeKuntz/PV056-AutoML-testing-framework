import argparse
import json
import os
import sys
import csv
from datetime import datetime
from hashlib import md5
from multiprocessing import Process, Queue

from pv056_2019.data_loader import DataLoader
from pv056_2019.schemas import ODStepConfigSchema, OutlierDetectorSchema
from pv056_2019.utils import valid_path, BASELINE_NAME, NONE_STR


class ODJobInfo:
    dataset: str
    fold: str
    setting: OutlierDetectorSchema
    hex: str
    input_filepath: str
    output_filepath: str

    def __init__(self, dataset: str, fold: str, setting: OutlierDetectorSchema, hex: str, input_filepath: str, output_filepath: str):
        self.dataset = dataset
        self.fold = fold
        self.setting = setting
        self.hex = hex
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

def od_worker(queue: Queue, times_file: str, backup_ts):
    while not queue.empty():
        od_job: ODJobInfo = queue.get()
        #od_settings = od_job.setting
        #train_file_path = od_job.input_filepath
        #file_save_path = od_job.output_filepath

        print(
            od_job.setting.name + ":",
            os.path.basename(od_job.input_filepath),
            "->",
            os.path.basename(od_job.output_filepath),
            flush=True,
        )

        try:
            dataframe = DataLoader._load_arff_file(od_job.input_filepath)

            # here we check if the OD setting is empty
            if od_job.setting.name == NONE_STR:
                # skip outlier detection and just dump it straight to the destination path
                od_frame = dataframe
                od_time = 0
            else:
                print("BEFORE " + od_job.setting.name, flush=True)
                od_frame, od_time = dataframe.apply_outlier_detector(od_job.setting)
                print("AFTER " + od_job.setting.name, flush=True)

            od_frame.arff_dump(od_job.output_filepath)
            print("DUMPED " + od_job.setting.name, flush=True)
            #print(od_job.input_filepath)
            #file_split = os.path.basename(od_job.output_filepath).split("_")


            with open(times_file, "a") as tf:
                print(od_job.dataset + "," + od_job.fold + "," + od_job.hex + "," + str(od_time), file=tf, flush=True)
            # here we are saving outputs to a second, unique file, which serves as a backup when overwriting previous file
            with open(backup_ts, "a") as tf:
                print(od_job.dataset + "," + od_job.fold + "," + od_job.hex + "," + str(od_time), file=tf, flush=True)
            print("TIMES WRITTEN " + od_job.setting.name, flush=True)
        except Exception as exc:
            print(
                "CAUGHT Error:\n\t{} {}\n\t".format(
                    od_job.setting.name, os.path.basename(od_job.input_filepath)
                ),
                exc,
                file=sys.stderr,
                flush=True,
            )
        except RuntimeWarning as any_warning:
            print("CAUGHT Runtime Warning:\n\t{} {}\n\t".format(
                    od_job.setting.name, os.path.basename(od_job.input_filepath)
                ),
                any_warning,
                file=sys.stderr,
                flush=True
            )
    print("QUEUE EMPTY")

def main():
    parser = argparse.ArgumentParser(
        description="Apply outlier detection methods to training data"
    )
    parser.add_argument("--config-file",
                        "-c",
                        required=True,
                        type=lambda x: valid_path(x, "Invalid path to configuration json file!"),
                        help="JSON configuration")
    parser.add_argument(
        "-do",
        "--datasets-csv-out",
        help="Path to a csv file which contains result files mappings and their new configurations histories, as modified by this step.",
        required=False,
    )
    parser.add_argument(
        "-di",
        "--datasets-csv-in",
        type=lambda x: valid_path(x, "Invalid path to input csv file!"),
        help="Path to csv file that contains previous data files mappings, locations and configurations.",
        required=True,
    )
    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        conf = ODStepConfigSchema(**json.load(json_file))

    #train_data_loader = DataLoader(conf.train_split_dir, regex=r".*_train\.arff")



    # here we write the headding of a times file
    with open(conf.times_output, "w") as tf:
        print("dataset,fold,od_hex,od_time", file=tf)
    backup_ts = "backups/" + os.path.basename(conf.times_output).replace(".csv", datetime.now()
                                                                         .strftime("_backup_%Y-%m-%d_%H-%M.csv"))
    # backup file form accidental overwrite
    with open(backup_ts, "w") as tf:
        print("dataset,fold,od_hex,od_time", file=tf)



    #with open(os.path.join(conf.train_od_dir, BASELINE_NAME + ".json"), "w") as out_config:
        #out_config.write('{"name": "None", "parameters": {}}')

    with open(args["datasets_csv_in"], "r") as datasets_csv_in:
        reader = csv.reader(datasets_csv_in, delimiter=",")
        # here we get an array of datasets.csv lines arranged by the size of the file in first column of each row
        datasets_rows = sorted([row for row in reader], key=lambda x: os.path.getsize(x[0]) if os.path.exists(x[0]) else 0)

    queue = Queue()
    # last considered setting is a baseline setting, which is basically empty
    # TODO: Remove limit to 30 datasets
    #counter = 0
    with open(args["datasets_csv_out"], "w", encoding="UTF-8") as output_csv:
        # TODO xbajger: Creating baseline files just for OD step is currently disabled. Considered useless.
        for od_settings in conf.od_methods: #+ [OutlierDetectorSchema(**{"name": NONE_STR, "parameters": {}})]:
            #if counter >= 30:
            #    pass  # continue
            # if the setting is empty (None) take it as baseline setting and mark it by appropriate hex string
            if od_settings.name == NONE_STR:
                hex_name = BASELINE_NAME
            else:
                hex_name = md5(od_settings.json(sort_keys=True).encode("UTF-8")).hexdigest()
            config_save_path = os.path.join(conf.train_od_dir, hex_name + ".json")
            with open(config_save_path, "w") as out_config:
                out_config.write(od_settings.json(sort_keys=True))

            for row in datasets_rows:
                train_file_path = row[0]
                test_file_path = row[1]
                previous_configs = row[2:]

                file_basename: str = os.path.basename(train_file_path)
                new_file_name = file_basename.replace(
                    "_train.arff", "_OD-" + hex_name + "_train.arff"
                )
                file_save_path = os.path.join(conf.train_od_dir, new_file_name)

                # TODO: extract dataset name and fold from SPLIT config (first json in csv)
                name_split = file_basename.split("_")
                dataset_name: str = name_split[0]
                fold = name_split[1]

                # write result paths into an output dataset ahead of actualy creating those files to avoid concurrent writing
                # between processes

                output_csv.write(",".join([file_save_path, test_file_path, *previous_configs, config_save_path]) + "\n")

                queue.put(ODJobInfo(
                    dataset=dataset_name,
                    fold=fold,
                    setting=od_settings,
                    hex=hex_name,
                    input_filepath=train_file_path,
                    output_filepath=file_save_path
                ))
                #counter += 1

    pool = [Process(target=od_worker, args=(queue, conf.times_output, backup_ts,)) for _ in range(conf.n_jobs)]

    # this is some medium level hacking. I am manually patching the sys.warnoptions then reverting it back
    old_warnopts = sys.warnoptions
    try:
        sys.warnoptions = []
        [process.start() for process in pool]
    except KeyboardInterrupt:
        [process.terminate() for process in pool]
        print("\nInterupted!", flush=True, file=sys.stderr)
    finally:
        # here I revert back the warn options
        sys.warnoptions = old_warnopts

    [process.join() for process in pool]
    print("Done")


if __name__ == "__main__":
    main()
