import argparse
import json
# TODO: Move the original definition into Utils (if possible)
import subprocess
import sys
from multiprocessing import Queue, Manager, Process

from pv056_2019.data_loader import DataLoader
from pv056_2019.main_clf import _valid_config_path
from pv056_2019.schemas import FeatureSelectionStepSchema
from pv056_2019.feature_selection.feature_evaluation import FeatureSelectionManager, _assert_trailing_slash, \
    FSCommandWithInfo




def extract_and_save_ranking_from_fs_output(fs_output: str, fs_output_filepath: str):

    #with open(fs_output_filepath, mode="w") as output_file:
    print(fs_output)


def fs_weka_worker(queue: Queue, blacklist: (str, str), timeout):
    while not queue.empty():
        command: FSCommandWithInfo = queue.get()
        args = command.args
        time_diff: float

        #file_split = args[6].split("/")[-1].split("_")
        dataset = command.dataset
        eval_method = command.eval_method_name

        if not (eval_method, dataset) in blacklist:
            try:
                #time_start = resource.getrusage(resource.RUSAGE_CHILDREN)[0]
                results = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
                #time_end = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

                extract_and_save_ranking_from_fs_output(results.stdout.decode(), command.output_file_path)


                #time_diff = time_end - time_start
            except subprocess.TimeoutExpired:
                time_diff = timeout
                blacklist.append((eval_method, dataset))
        else:
            time_diff = timeout



        #clf_fam = ".".join(args[16].split(".")[2:-1])
        #fs_hex = args[10].split("/")[-1].split("_")[-2]
        #fold = command.fold
        #od_hex = file_split[2]
        #rm = file_split[3].split("-")[1]

        #with open(times_file, "a") as tf:
        #    print(",".join([dataset, fold, eval_method, fs_hex, od_hex, rm, str(time_diff)]), file=tf)


        #print(";".join([args[16], args[6], args[8]]), flush=True)


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
        "--fs-mapping-csv-out",
        help="path to a csv file which contains datasets used for FS and their respective result files",
        required=False,
    )

    parser.add_argument(
        "-d",
        "--datasets-csv-in",
        type=_valid_config_path,
        help="Path to csv file that contains previous data files mappings, locations and for example OD configurations",
        required=True,
    )
    args = parser.parse_args()

    # load config file
    with open(args.config_fs, "r") as config_file:
        conf = FeatureSelectionStepSchema(**json.load(config_file))

    # TODO: Here implement getting filepaths from datasets.csv instead of temporary hardcoded list of files
    #dataset_paths = ["data/datasets/eye-movements.arff",
    #         "data/datasets/glass.arff",
    #         "data/datasets/heart-c.arff",
    #         "data/datasets/hepatitis.arff"
    #         ]
    #fs_data_loader = DataLoader(conf.input_folder_path, regex=r".*_train\.arff")

    #datasets_output = []

    # try:
    #     print("Removing {}%".format(conf.percentage), flush=True)
    #
    #     processed_base = []
    #     for dataframe, train_file_path in zip(
    #             fs_data_loader.load_files(), fs_data_loader.file_paths
    #     ):

    fs_manager = FeatureSelectionManager(conf)
    # default path to a CSV with datasets and their FS results is with the results
    fs_mapping_csv_path = "fs_mapping.csv" if args.fs_mapping_csv_out is None else args.fs_mapping_csv_out

    #WEKA_commands = []


    #print(WEKA_commands)

    # this manager allows us to make synchronized data structures
    with Manager() as manager:
        # here we create a list, that will be synchronize across threads, so we can modify it in between processes
        blacklist = manager.list()
        with open(conf.blacklist_file_path, "r") as bf:
            for i in bf:
                blacklist.append(i.replace("\n", "").split(','))

        queue = Queue()
        with open(fs_mapping_csv_path, "w") as fs_mapping_csv, open(args.datasets_csv_in, "r") as datasets_mapping_csv:
            # Here we open a file to record which file was used for feature selection, where the result was stored and which
            # parameters were used in the process (config).
            # The file is then passed to the method below to write the actual mappings, because it is easier.
            # the generator below yields commands but also records their arguments to a csv file for reference
            for command in fs_manager.generate_fs_weka_commands(datasets_mapping_csv, fs_mapping_csv):
                print(" ".join(command.args))
                queue.put(command)



        # create a pool of processes that will work in parallel
        pool = [Process(target=fs_weka_worker, args=(queue, blacklist, conf.timeout,)) for _ in range(conf.n_jobs)]

        try:
            [process.start() for process in pool]
            # join below will result in waiting for all above processes to finish before continuing
            [process.join() for process in pool]
        except KeyboardInterrupt:
            [process.terminate() for process in pool]
            print("\nInterupted!", flush=True, file=sys.stderr)

    print("Done")

if __name__ == '__main__':
    main()