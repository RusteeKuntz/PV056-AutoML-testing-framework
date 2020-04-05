import argparse
import csv
import json
import os
import sys
import random

from sklearn.model_selection import KFold

from pv056_2019.data_loader import DataLoader
from pv056_2019.schemas import SplitterSchema


def main():
    parser = argparse.ArgumentParser(
        description="Script splits datasets for m*k-fold cross-validation. Script splits datasets m-times into k folds."
    )
    parser.add_argument("--config-file", "-c", required=True, help="JSON configuration")
    parser.add_argument(
        "--datasets-file",
        "-do",
        required=True,
        help="Filename of output datasets csv tracking files relationships",
    )

    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        conf = SplitterSchema(**json.load(json_file))

    data_loader = DataLoader(conf.data_path)

    # if number of k-fold repeats is one, behaviour is same as in previous versions
    if conf.m_of_repeats > 1:
        # if number of repeats is greater than 1, only then initialize random values for splitting.
        # For compatibility purposes
        r = random.Random(conf.random_state)
        # this way the random states cannot repeat (the measure is probably not necessary, but safe is safe)
        # random states have to be integers, or numpy RandomState instances
        randomness_space = 10000
        pseudo_random_states = [r.randint(randomness_space * m, randomness_space * (m+1)) for m in range(conf.m_of_repeats)]
    else:
        pseudo_random_states = [conf.random_state]

    datasets_output = []
    try:
        for dataframe in data_loader.load_files():
            print("Splitting:", dataframe._arff_data["relation"], flush=True)
            dataframe = dataframe.add_index_column()
            for m in range(conf.m_of_repeats):

                # previously, in early versions, the random_state here was 42. This number was moved into SplitterSchema
                # datasets_file a default value, when AdamBajger was upgrading the splitting method.
                kfold = KFold(n_splits=conf.k_of_folds, shuffle=True, random_state=pseudo_random_states[m])
                for k, data_fold in enumerate(kfold.split(dataframe.index.values)):
                    train_index, test_index = data_fold

                    train_frame = dataframe.select_by_index(train_index)
                    train_name = (
                        dataframe._arff_data["relation"] + "_" + str(k) + "-" + str(m) + "_train.arff"
                    )
                    train_split_output = os.path.join(conf.train_split_dir, train_name)
                    train_frame.arff_dump(train_split_output)

                    test_frame = dataframe.select_by_index(test_index)
                    test_name = (
                        dataframe._arff_data["relation"] + "_" + str(k) + "-" + str(m) + "_test.arff"
                    )
                    test_split_output = os.path.join(conf.test_split_dir, test_name)
                    test_frame.arff_dump(test_split_output)

                    datasets_output.append([train_split_output, test_split_output, ""])

    except KeyboardInterrupt:
        print("\nInterupted!", flush=True, file=sys.stderr)

    with open(args["datasets_file"], "w") as datasets_file:
        writer = csv.writer(datasets_file, delimiter=",")
        writer.writerows(datasets_output)

    print("Done")


if __name__ == "__main__":
    main()
