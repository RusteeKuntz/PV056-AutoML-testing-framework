import argparse
import csv
import json
import os
import sys

from pv056_2019.data_loader import DataLoader
from pv056_2019.schemas import RemoveOutliersConfigSchema
from pv056_2019.utils import OD_VALUE_NAME


def main():
    parser = argparse.ArgumentParser(
        description="Removes the percentage of the largest outliers."
    )
    parser.add_argument("--config-file", "-c", required=True, help="JSON configuration")
    parser.add_argument(
        "--datasets-file",
        "-d",
        required=True,
        help="Filename of output datasets config",
    )

    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        conf = RemoveOutliersConfigSchema(**json.load(json_file))

    train_data_loader = DataLoader(conf.train_od_dir, regex=r".*_train\.arff")

    datasets_output = []

    try:
        print("Removing {}%".format(conf.percentage), flush=True)

        processed_base = []
        for dataframe, train_file_path in zip(
            train_data_loader.load_files(), train_data_loader.file_paths
        ):
            if OD_VALUE_NAME not in dataframe.columns:
                print(
                    "Skipping {}. File does not have an OD_VALUE.".format(
                        train_file_path
                    ),
                    flush=True,
                    file=sys.stderr,
                )
                continue

            percentages = (
                conf.percentage
                if isinstance(conf.percentage, list)
                else [conf.percentage]
            )
            if conf.keep_original and not 0 in percentages:
                    percentages.append(0)


            for percentage in percentages:
                # check if base output
                ds_fold_pair = os.path.basename(train_file_path).split("_")[:2]
                if percentage == 0:
                    if not ds_fold_pair in processed_base:
                        processed_base.append(ds_fold_pair)
                    else:
                        continue
                try:
                    # here the outliers are removed
                    new_frame = dataframe.select_by_od_quantile(1 - (percentage / 100), conf.reverse)
                    new_frame.pop(OD_VALUE_NAME)
                    new_frame._arff_data["attributes"] = [
                        x
                        for x in new_frame._arff_data["attributes"]
                        if x[0] != OD_VALUE_NAME
                    ]
                    print("   ", train_file_path, "{}%".format(percentage))
                    name_split = os.path.basename(train_file_path).split("_")
                    if percentage == 0:
                        name_split[-2] = "baseline"
                    name_split.insert(-1, "removed-{0:.2f}".format(round(percentage, 2)))

                    file_name = "_".join(name_split)
                    file_save_path = os.path.join(conf.train_removed_dir, file_name)

                    new_frame.arff_dump(file_save_path)

                    # here we keep datasets mapping. We write it at the end to a CSV
                    # the location of a test split is predicted based on a file name. How disgusting.
                    # TODO xbajger: This really has to be reworked so that it uses a datasets.csv as all other steps.
                    datasets_output.append(
                        [
                            file_save_path,
                            os.path.join(
                                conf.test_split_dir,
                                "_".join(name_split[:2]) + "_test.arff",
                            ),
                            os.path.join(conf.train_od_dir, name_split[2] + ".json"),
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
    with open(args["datasets_file"], "w") as datasets_file:
        writer = csv.writer(datasets_file, delimiter=",")
        writer.writerows(datasets_output)

    print("Done")


if __name__ == "__main__":
    main()
