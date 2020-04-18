import json
import os
import hashlib
import re
from multiprocessing import Queue

from pv056_2019.utils import ID_NAME, OD_VALUE_NAME

from pv056_2019.schemas import ClassifierSchema
from typing import List
from itertools import product


class ClassifierManager:

    # Weka classifiers
    # http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html

    # java -cp weka.jar weka.classifiers.meta.FilteredClassifier
    # -t data/diabetes.arff
    # -F "weka.filters.unsupervised.attribute.RemoveByName -E ^ID$"
    # -x 5 -S 1
    # -W weka.classifiers.trees.J48 -- -C 0.25 -M 2

    def __init__(self, output_folder, weka_jar_path):
        self.output_folder = output_folder
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)
        self.weka_jar_path = weka_jar_path
        if not os.path.exists(self.weka_jar_path):
            raise IOError(
                "Input weka.jar file, '{0}' does not exist.".format(self.weka_jar_path)
            )

        self._regex_removed = re.compile(r"_removed-\d+\.\d+")


    @staticmethod
    def _create_final_config_file(dataset_conf_paths, classifier):
        """This method returns the final full configuration of an experiment in the form of JSON string"""
        json_configs = []
        for dataset_conf_path in dataset_conf_paths:
            with open(dataset_conf_path, "r") as f:
                json_configs.append(json.load(f))

        final_config = json.dumps(
            {"model_config": classifier.dict(), "od_configs": json_configs},
            indent=4,
            separators=(",", ":"),
        )
        return final_config

    @staticmethod
    def _save_model_config(config_file_path, config_data):
        with open(config_file_path, "w") as f:
            f.write(config_data)

    def fill_queue_and_create_configs(
        self,
        queue: Queue,
        classifiers: List[ClassifierSchema],
        dataset_tuples: List[List[str]],
    ):
        for dataset_tuple, classifier in product(dataset_tuples, classifiers):
            # first two elements the datase_tuple are train file and test file
            train_path, test_path = dataset_tuple[:2]
            conf_paths = dataset_tuple[2:]

            if not os.path.exists(train_path):
                raise IOError("Input dataset '{0}' does not exist.".format(train_path))

            # Get the string with final configuration of the whole workflow
            final_config_str = self._create_final_config_file(conf_paths, classifier)
            # get identifying hash
            hash_md5 = hashlib.md5(final_config_str.encode()).hexdigest()

            basename: str = os.path.basename(train_path)
            basename_split: [str] = basename.split("_")
            dataset_name = basename_split[0]
            dataset_fold = basename_split[1]

            removed_arr = self._regex_removed.findall(basename)
            if removed_arr:
                removed_str = removed_arr[0]
            else:
                removed_str = ""

            #this is the filepath for the prediction output from trained WEKA classifier
            predict_file_path = os.path.join(
                self.output_folder,
                dataset_name
                + "_"
                + dataset_fold
                + "_"
                + classifier.name
                + "_"
                + hash_md5
                + removed_str
                + ".csv",
            )
            # this is the filepath for a JSON with full configuration of an experiment
            config_file_path = os.path.join(
                self.output_folder, classifier.name + "_" + hash_md5 + ".json"
            )

            # Prepare arguments for classifier
            run_args: List[str] = []
            run_args += ["-t", train_path]  # input dataset
            run_args += ["-T", test_path]  # input dataset
            run_args += [
                "-classifications",
                "weka.classifiers.evaluation.output.prediction.CSV -p first -file {0} -suppress".format(  # noqa
                    predict_file_path
                ),
            ]

            # Add Weka filters
            # here we filter out OD column and INDEX column
            str_filters = '-F "weka.filters.unsupervised.attribute.RemoveByName -E ^{}$"'.format(  # noqa
                ID_NAME
            ) + ' -F "weka.filters.unsupervised.attribute.RemoveByName -E ^{}$"'.format(
                OD_VALUE_NAME
            )
            for one_filter in classifier.filters:
                str_filters += '-F "{0} {1}"'.format(
                    one_filter.name, " ".join(one_filter.args)
                )
            run_args += ["-F", "weka.filters.MultiFilter {0}".format(str_filters)]

            run_args += ["-S", "1"]  # Seed
            run_args += ["-W", classifier.class_name]
            if classifier.args:
                run_args += ["--"]
                run_args += classifier.args

            run_args = [
                "java",
                "-Xmx1024m",
                "-cp",
                self.weka_jar_path,
                "weka.classifiers.meta.FilteredClassifier",
            ] + run_args

            command = CLFCommandWithInfo(args=run_args, dataset_name=dataset_name, train_path=train_path, clf=classifier.class_name, fold=dataset_fold, settings=final_config_str)

            queue.put(command)
            self._save_model_config(config_file_path, final_config_str)


class CLFCommandWithInfo:
    """
    This class encapsulates CLF command args for WEKA to be run from commandline via python subporccess
    and also any further information needed so that we dont have to extract it back from
    the args manually. It helps with code readability and allows easier project management.
    """
    args: [str]
    dataset: str
    train_path: str
    clf_classname: str
    fold: str
    settings: [str]

    def __init__(self,
                 args: [str],
                 dataset_name: str,
                 train_path: str,
                 clf: str,
                 fold: str,
                 settings: [str]
                 ):
        self.args = args
        self.dataset = dataset_name
        self.train_path = train_path
        self.clf_classname = clf
        self.fold = fold
        self.settings = settings