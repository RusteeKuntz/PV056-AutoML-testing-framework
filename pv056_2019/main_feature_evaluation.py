import argparse, os
import json
# TODO: Move the original definition into Utils (if possible)
import subprocess
import sys, resource
from multiprocessing import Queue, Manager, Process
from typing import TextIO

from pv056_2019.data_loader import DataLoader, DataFrameArff
from pv056_2019.feature_selection import setup_sklearn_fs_class
from pv056_2019.main_clf import _valid_config_path
from pv056_2019.schemas import FeatureSelectionStepSchema, ScikitFSSchema
from pv056_2019.feature_selection.feature_evaluation import FeatureSelectionManager, FSJobWithInfo
from pv056_2019.utils import valid_path, CUSTOM, SCIKIT

debugging = False


def run_scikit_stuff(args: ScikitFSSchema, df: DataFrameArff, output_file_path: str):

    print("Selection began for", df.arff_data().relation)
    fs_frame = df.select_features_with_sklearn(
        setup_sklearn_fs_class(args.fs_method, args.score_func),
        args.leave_attributes_binarized
    )

    fs_frame.arff_dump(output_file_path)
    print("Dump completed.")


def fs_worker(queue: Queue, mapping_csv: TextIO, blacklist: [(str, str)], binarized_test_files, timeout):
    while not queue.empty():
        command: FSJobWithInfo = queue.get()
        time_diff: float

        dataset = os.path.basename(command.input_path).split("_")[0]
        eval_method = command.eval_method_name

        print("Popped command for", eval_method, "-->", dataset, "from queue.")

        # I believe it does not make sense to blacklist search methods.
        print("FSBLACKLIST\n", blacklist, "\nEVAL METHOD and DATASET: ", (eval_method, dataset))
        if not (eval_method, dataset) in blacklist:
            if command.is_cmd:
                try:
                    time_start = resource.getrusage(resource.RUSAGE_CHILDREN)[0]
                    results = subprocess.run(command.args,
                                             #capture_output=True,  # this is not supported in current version
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             timeout=timeout)

                    # This is POSIX dependent piece of code
                    if results.returncode < 0:
                        print(eval_method, dataset, "TERMINATED externally. Skipping")
                        continue
                    #print("ERR:", eval_method, dataset, results.stderr)
                    #print("DONE?")
                    # print("RESULTS OMFG")
                    # print(results.args)
                    # print("OUT")
                    # print(results.stdout.decode(encoding="UTF-8"))
                    # print("ERR")
                    # print(results.stderr.decode(encoding="UTF-8"))
                    time_end = resource.getrusage(resource.RUSAGE_CHILDREN)[0]
                    time_diff = time_end - time_start
                except subprocess.TimeoutExpired:
                    print(eval_method, dataset, "reached the timeout. TERMINATED on TIMEOUT")
                    time_diff = timeout
                    blacklist.append((eval_method, dataset))
                    continue  # continue on timeout
            else:
                #print("We are in a correct conditional branch, loading file on path: " + command.input_path)
                df = DataLoader._load_arff_file(command.input_path)
                if command.args.source_library == CUSTOM:
                    #print("We are in a wrong conditional branch")
                    fs_frame, time_diff = df.apply_custom_feature_selector(command.args)
                elif command.args.source_library == SCIKIT:
                    # print("We are in a correct conditional branch 2")
                    args: ScikitFSSchema = command.args

                    # if we are leaving the categorical attributes binarized, we need to binarize the test split too
                    if args.leave_attributes_binarized:

                        # here we extract the test filepath of the test split
                        csv_line_split = command.mapping_csv_line.split(",")
                        test_file_path = csv_line_split[1]

                        # here we create new filepath
                        test_file_path_dotsplit = test_file_path.split(".")
                        bin_test_file_path = ".".join(test_file_path_dotsplit[0:-1]) \
                                             + "_bin." + test_file_path_dotsplit[-1]
                        if bin_test_file_path not in binarized_test_files:
                            print("Binarizing test data for", eval_method, dataset)
                            test_df = DataLoader._load_arff_file(test_file_path)
                            bin_test_df = test_df.binarize_cat_feats_and_normalize(keep_class=True)

                            # Here we dump the binarized test dataframe to a that path
                            bin_test_df.arff_dump(bin_test_file_path)
                            binarized_test_files[bin_test_file_path] = True
                        else:
                            print("Test files already binarized for", eval_method, dataset)
                        # here we overwrite the old test path for this particular file
                        # (old path is still used in non-binarized datasets)
                        command.mapping_csv_line = ",".join(
                            [csv_line_split[0], bin_test_file_path, *csv_line_split[2:]])

                    time_start = resource.getrusage(resource.RUSAGE_SELF)[0]
                    time_start_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

                    # Start bar as a process
                    p = Process(target=run_scikit_stuff, args=(args, df, command.output_file_path))
                    p.start()

                    # Wait for "timeout" seconds or until process finishes
                    p.join(timeout=timeout)

                    # If thread is still active
                    if p.is_alive():
                        print(eval_method, dataset, "reached the timeout, killing it...")

                        # Terminate
                        p.terminate()
                        p.join()
                        fs_time = timeout
                        print(eval_method, dataset, "TERMINATED on TIMEOUT")
                        continue  # if timeout,
                    elif p.exitcode < 0:
                        print(eval_method, dataset, "TERMINATED externally with exitcode", p.exitcode, "Skipping")
                        continue


                    # conclude time (resource) consumption if proccess finished properly
                    time_end = resource.getrusage(resource.RUSAGE_SELF)[0]
                    time_end_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]
                    fs_time = (time_end - time_start) + (time_end_children - time_start_children)
                    # #print("We are in a correct conditional branch 2")
                    # args: ScikitFSSchema = command.args
                    #
                    # # if we are leaving the categorical attributes binarized, we need to binarize the test split too
                    # if args.leave_attributes_binarized:
                    #     # here we extract the test filepath of the test split
                    #     csv_line_split = command.mapping_csv_line.split(",")
                    #     test_file_path = csv_line_split[1]
                    #     test_df = DataLoader._load_arff_file(test_file_path)
                    #     bin_test_df = test_df.binarize_cat_feats_and_normalize(keep_class=True)
                    #
                    #     # here we create new filepath and dump the binarized test dataframe to a that path
                    #     test_file_path_dotsplit = test_file_path.split(".")
                    #     bin_test_file_path = ".".join(test_file_path_dotsplit[0:-1])\
                    #                          + "_bin." + test_file_path_dotsplit[-1]
                    #     bin_test_df.arff_dump(bin_test_file_path)
                    #     # here we overwrite the old test path for this particular file
                    #     # (old path is still used in non-binarized datasets)
                    #     command.mapping_csv_line = ",".join(
                    #         [csv_line_split[0], bin_test_file_path, *csv_line_split[2:]])
                    #
                    # fs_frame, time_diff = df.select_features_with_sklearn(
                    #     setup_sklearn_fs_class(args.fs_method, args.score_func),
                    #     args.leave_attributes_binarized
                    # )
                else:
                    raise NotImplementedError()

            # log the finish of specific command
            print("Completed: ", eval_method, dataset)
            #fs_frame.arff_dump(command.output_file_path)

            # We write the line into the datasets mapping CSV only if the preprocessing is actually done
            #print("The line printed is:", command.mapping_csv_line, sep="\n")
            mapping_csv.write(command.mapping_csv_line)
            mapping_csv.flush()

        else:
            # previously timed out combination is skipped right away
            time_diff = timeout

        # TODO: Add times output for feature selection
        # clf_fam = ".".join(args[16].split(".")[2:-1])
        # fs_hex = args[10].split("/")[-1].split("_")[-2]
        # fold = command.fold
        # od_hex = file_split[2]
        # rm = file_split[3].split("-")[1]

        # with open(times_file, "a") as tf:
        #    print(",".join([dataset, fold, eval_method, fs_hex, od_hex, rm, str(time_diff)]), file=tf)

        # print(";".join([args[16], args[6], args[8]]), flush=True)


def main():
    parser = argparse.ArgumentParser(description="PV056-AutoML-testing-framework")
    parser.add_argument(
        "-c",
        "--config-fs",
        type=lambda x: valid_path(x, "Invalid path to configuration json file!"),
        help="path to feature selection config file",
        required=True,
    )
    parser.add_argument(
        "-do",
        "--datasets-csv-out",
        help="path to a csv file which contains datasets used for FS and their respective result files",
        required=False,
    )
    parser.add_argument(
        "-di",
        "--datasets-csv-in",
        type=lambda x: valid_path(x, "Invalid path to input csv file!"),
        help="Path to csv file that contains previous data files mappings, locations and for example OD configurations",
        required=True,
    )
    args = parser.parse_args()

    print("Starting feature selection step")

    # load config file
    with open(args.config_fs, "r") as config_file:
        conf = FeatureSelectionStepSchema(**json.load(config_file))

    fs_manager = FeatureSelectionManager(conf)
    # default path to a CSV with datasets and their FS results is with the results
    fs_mapping_csv_path = "fs_mapping.csv" if args.datasets_csv_out is None else args.datasets_csv_out

    # this manager allows us to make synchronized data structures
    with Manager() as manager:
        # here we create a list, that will be synchronize across threads, so we can modify it in between processes
        blacklist = manager.list()
        binarized_test_files = manager.dict()
        if conf.blacklist_file_path is not None:
            with open(conf.blacklist_file_path, "r") as bf:
                for i in bf:
                    blacklist.append(tuple(i.replace("\n", "").split(',')))

        queue = Queue()
        with open(args.datasets_csv_in, "r") as datasets_mapping_csv:
            # Here we open a file to record which file was used for feature selection, where the result was stored and which
            # parameters were used in the process (config).
            # The file is then passed to the method below to write the actual mappings, because it is easier.
            # the generator below yields commands but also records their arguments to a csv file for reference
            for command in fs_manager.generate_fs_jobs(datasets_mapping_csv):
                # print(" ".join(command.args))
                queue.put(command)

        print("queue filled")

        # get a file handle to write preprocessed datasets and their configuration histories to. File gets closed later
        fs_mapping_csv = open(fs_mapping_csv_path, "w", encoding="UTF-8")
        # create a pool of processes that will work in parallel
        pool = [Process(target=fs_worker, args=(queue, fs_mapping_csv, blacklist, binarized_test_files, conf.timeout)) for _ in
                range(conf.n_jobs)]

        print("Starting processes")
        try:
            [process.start() for process in pool]
            print("Processes started, joining them...", flush=True)
            # join below will result in waiting for all above processes to finish before continuing
            [process.join() for process in pool]
        except KeyboardInterrupt:
            [process.terminate() for process in pool]
            print("\nInterupted!", flush=True, file=sys.stderr)
        # close the mapping file after everything is finished
        fs_mapping_csv.close()

    print("Done")


if __name__ == '__main__':
    main()
