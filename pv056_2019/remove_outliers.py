import argparse
import csv
import json
import os
import sys

from pv056_2019.data_loader import DataLoader
from pv056_2019.schemas import RemoveOutliersConfigSchema
from pv056_2019.utils import OD_VALUE_NAME, valid_path, BASELINE_NAME


def main():
    parser = argparse.ArgumentParser(
        description="Removes the percentage of the largest outliers."
    )
    parser.add_argument("--config-file", "-c", required=True, type=lambda x: valid_path(x, "Invalid config json path!"),
                        help="JSON configuration")
    parser.add_argument(
        "--datasets-csv-in",
        "-di",
        required=True,
        type=lambda x: valid_path(x, "Invalid input csv path!"),
        help="CSV file with input dataset split mappings and configuration histories",
    )
    parser.add_argument(
        "--datasets-csv-out",
        "-do",
        required=True,
        help="CSV containing resulting dataset split mappings and updated configuration histories",
    )
    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        conf: RemoveOutliersConfigSchema = RemoveOutliersConfigSchema(**json.load(json_file))
    #train_data_loader = DataLoader(conf.train_od_dir, regex=r".*_train\.arff")

    datasets_output = []
    try:
        print("Removing {}%".format(conf.percentage), flush=True)

        processed_base = []
        with open(args["datasets_csv_in"], "r") as csv_in:
            reader = csv.reader(csv_in)
            for row in reader:
                train_file_path = row[0]
                test_file_path = row[1]
                configs_history = row[2:]
                preceeding_config_json = configs_history[-1]
                preceeding_hex = os.path.basename(preceeding_config_json).split(".")[0]
                if os.path.exists(train_file_path):
                    dataframe = DataLoader._load_arff_file(train_file_path)
                else:
                    print("File {} does not exist and was skipped.".format(train_file_path))
                    continue

                if preceeding_hex == BASELINE_NAME:
                    percentages = [0]
                else:
                    # skip non-baseline files that have no OD_VALUES column
                    if OD_VALUE_NAME not in dataframe.columns:
                        print(
                            "Skipping {}. File does not have an OD_VALUE.".format(
                                train_file_path
                            ),
                            flush=True,
                            file=sys.stderr,
                        )
                        continue

                    # if percentage is a value (not list) turn it into a list with one value
                    percentages = (
                        conf.percentage
                        if isinstance(conf.percentage, list)
                        else [conf.percentage]
                    )

                # if conf.keep_original and not 0 in percentages:
                #         percentages.append(0)
                for percentage in percentages:
                    # check if base output
                    # ds_fold_pair = os.path.basename(train_file_path).split("_")[:2]
                    # if percentage == 0:
                    #     if not ds_fold_pair in processed_base:
                    #         processed_base.append(ds_fold_pair)
                    #     else:
                    #         continue
                    try:
                        if percentage == 0:
                            new_frame = dataframe
                        else:
                            # here the outliers are removed
                            rev = True if percentage < 0 else False
                            new_frame = dataframe.select_by_od_quantile(1 - (percentage / 100), rev)
                            new_frame.pop(OD_VALUE_NAME)
                            new_frame._arff_data["attributes"] = [
                                x
                                for x in new_frame._arff_data["attributes"]
                                if x[0] != OD_VALUE_NAME
                            ]
                        print("   ", train_file_path, "{}%".format(percentage))
                        file_basename = os.path.basename(train_file_path)
                        #name_split.insert(-1, "removed-{0:.2f}".format(round(percentage, 2)))

                        rm_setting = {"RM": percentage}
                        # unlike in other steps, here the setting is unique by the percent number, so no hex is needed
                        identifier = "RM{0:.2f}".format(round(percentage, 2))
                        # we probably do not need the setting to be saved too many times on so many places, but for the sake
                        # of keeping the structure similar, we do it here the same way,
                        # save config to json and keep its path in the CSV
                        conf_save_path = os.path.join(conf.train_removed_dir, identifier + ".json")
                        with open(conf_save_path, "w", encoding="UTF-8") as conf_json:
                            json.dump(obj=rm_setting, fp=conf_json)
                        new_file_name = file_basename.replace(
                            "_train.arff", "_" + identifier + "_train.arff"
                        )

                        file_save_path = os.path.join(conf.train_removed_dir, new_file_name)

                        new_frame.arff_dump(file_save_path)

                        # here we keep datasets mapping. We write it at the end to a CSV
                        datasets_output.append(
                            [
                                file_save_path,
                                test_file_path,
                                *configs_history, # python unpacking of list elements
                                conf_save_path
                            ]
                        )
                    except KeyboardInterrupt as keyb:
                        raise keyb
                    except Exception as exc:
                        print(
                            "Error:",
                            train_file_path,
                            "{}%".format(percentage),
                            exc,
                            file=sys.stderr,
                        )

    except KeyboardInterrupt:
        print("\nInterupted!", flush=True, file=sys.stderr)

    # Here we write the datasets mapping
    with open(args["datasets_csv_out"], "w") as datasets_file:
        writer = csv.writer(datasets_file, delimiter=",")
        writer.writerows(datasets_output)

    print("Done")


if __name__ == "__main__":
    main()
