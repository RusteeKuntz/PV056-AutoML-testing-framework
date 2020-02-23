import argparse
import json
import os
import sys
from datetime import datetime
from hashlib import md5
from multiprocessing import Process, Queue

from pv056_2019.data_loader import DataLoader
from pv056_2019.schemas import ODStepConfigSchema


def od_worker(queue: Queue, times_file: str, backup_ts):
    while not queue.empty():
        od_settings, train_file_path, file_save_path = queue.get()
        print(
            od_settings.name + ":",
            os.path.basename(train_file_path),
            "->",
            os.path.basename(file_save_path),
            flush=True,
        )

        try:
            dataframe = DataLoader._load_arff_file(train_file_path)

            od_frame, od_time = dataframe.apply_outlier_detector(od_settings)
            od_frame.arff_dump(file_save_path)

            print(train_file_path)
            file_split = os.path.basename(file_save_path).split("_")

            with open(times_file, "a") as tf:
                print(file_split[0] + "," + file_split[1] + "," + file_split[2] + "," + str(od_time), file=tf)
            with open(backup_ts, "a") as tf:
                print(file_split[0] + "," + file_split[1] + "," + file_split[2] + "," + str(od_time), file=tf)

        except Exception as exc:
            print(
                "Error:\n\t{} {}\n\t".format(
                    od_settings.name, os.path.basename(train_file_path)
                ),
                exc,
                file=sys.stderr,
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Apply outlier detection methods to training data"
    )
    parser.add_argument("--config-file", "-c", required=True, help="JSON configuration")

    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        conf = ODStepConfigSchema(**json.load(json_file))

    train_data_loader = DataLoader(conf.train_split_dir, regex=r".*_train\.arff")

    queue = Queue()

    with open(conf.times_output, "w") as tf:
        print("dataset,fold,od_hex,od_time", file=tf)
    backup_ts = "backups/" + os.path.basename(conf.times_output).replace(".csv", datetime.now()
                                                                      .strftime("_backup_%d-%m-%Y_%H-%M.csv"))
    with open(backup_ts, "w") as tf:
        print("dataset,fold,od_hex,od_time", file=tf)

    with open(os.path.join(conf.train_od_dir, "baseline.json"), "w") as out_config:
        out_config.write('{"name": "None", "parameters": {}}')

    for od_settings in conf.od_methods:
        hex_name = md5(od_settings.json(sort_keys=True).encode("UTF-8")).hexdigest()
        config_save_path = os.path.join(conf.train_od_dir, hex_name + ".json")
        with open(config_save_path, "w") as out_config:
            out_config.write(od_settings.json(sort_keys=True))

        for train_file_path in train_data_loader.file_paths:
            file_basename = os.path.basename(train_file_path)
            file_name = file_basename.replace(
                "_train.arff", "_" + hex_name + "_train.arff"
            )
            file_save_path = os.path.join(conf.train_od_dir, file_name)

            queue.put([od_settings, train_file_path, file_save_path])

    pool = [Process(target=od_worker, args=(queue, conf.times_output, backup_ts,)) for _ in range(conf.n_jobs)]

    try:
        [process.start() for process in pool]
        [process.join() for process in pool]
    except KeyboardInterrupt:
        [process.terminate() for process in pool]
        print("\nInterupted!", flush=True, file=sys.stderr)

    print("Done")


if __name__ == "__main__":
    main()
