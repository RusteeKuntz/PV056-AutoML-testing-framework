from pv056_2019.utils import get_datetime_now_str, get_clf_name,\
    yield_classifiers
import subprocess
import copy
import json
import os
import random


class ClassifierManager:

    # Weka classifiers
    # http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html

    # java -cp weka.jar weka.classifiers.meta.FilteredClassifier
    # -t data/diabetes.arff
    # -F "weka.filters.unsupervised.attribute.RemoveByName -E ^ID$"
    # -x 5 -S 1
    # -W weka.classifiers.trees.J48 -- -C 0.25 -M 2

    def __init__(self, log_folder, weka_jar_path):
        self.log_folder = log_folder
        if not os.path.isdir(self.log_folder):
            os.mkdir(self.log_folder)
        self.weka_jar_path = weka_jar_path
        if not os.path.exists(self.weka_jar_path):
            raise IOError(
                "Input weka.jar file, '{0}' does not exist.".format(self.weka_jar_path)
            )

    @staticmethod
    def _save_stds(stdout_file_path, stderr_file_path, output, errors, rc):
        with open(stdout_file_path, "w") as f:
            f.write(output)

        with open(stderr_file_path, "w") as f:
            f.write(errors)

    @staticmethod
    def _save_model_config(config_file_path, dataset_conf_path,
                           clf_class, clf_args):
        with open(dataset_conf_path, "r") as f:
            json_str = json.load(f)

        with open(config_file_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "model_config":{
                            "class_name": clf_class,
                            "args": clf_args,
                        },
                        "ad_config": json_str
                    },
                    indent=4,
                    separators=(",", ":"),
                )
            )

    @staticmethod
    def _print_run_info(clf_class, run_args):
        print("-" * 40)
        print("Running '{0}' with:".format(clf_class))
        for arg in run_args:
            arg_msg = " " + arg + "\n" if arg[0] != "-" else "\t" + arg
            print(arg_msg, end="")

    @staticmethod
    def run_subprocess(run_args):
        p = subprocess.Popen(run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
        rc = p.returncode
        return output.decode('UTF-8'), err.decode('UTF-8'), rc

    """
    def create_run_folder(self):
        run_folder_name = get_datetime_now_str() + "_run"
        folder = os.path.join(self.log_folder, run_folder_name)
        os.makedirs(folder, exist_ok=True)
        self.run_folder = folder

    def create_hash_folder(self, dataset_path, dataset_conf_path):
        hash_md5 = calculate_dataset_hash(dataset_path, dataset_conf_path)

        # Create folders
        folder = os.path.join(self.run_folder, hash_md5)
        os.makedirs(folder, exist_ok=True)
        return folder
    """

    def run(self, classifiers, dataset_tuples):
        # Run all classifiers with all datasets
        #self.create_run_folder()
        for clf_name, clf_args in yield_classifiers(classifiers):
            for dataset_tuple in dataset_tuples:
                self.run_weka_classifier(clf_name, clf_args, dataset_tuple)

    def run_weka_classifier(self, clf_class, clf_args, dataset_tuple):
        dataset_path = dataset_tuple[0]
        dataset_conf_path = dataset_tuple[1]

        # Check dataset path
        if not os.path.exists(dataset_path):
            raise IOError("Input dataset '{0}' does not exist.".format(dataset_path))

        # Prepare output folders
        #log_folder = self.create_hash_folder(dataset_path, dataset_conf_path)

        # Check clf_args
        if "-t" in clf_args or "-x" in clf_args:
            print("Settings '-t', '-x' will be overwritten.")

        # Create log_file names
        rand_int = random.randint(0, 1000)
        file_prefix = get_datetime_now_str() + "_" +\
            str(rand_int) + "_" + get_clf_name(clf_class)

        predict_file_path = os.path.join(self.log_folder, file_prefix + '.csv')
        config_file_path = os.path.join(self.log_folder, file_prefix + '.json')
        #stdout_file_path = os.path.join(log_folder, file_prefix + 'stdout.txt')
        #stderr_file_path = os.path.join(log_folder, file_prefix + 'stderr.txt')

        # Prepare arguments for classifier
        run_args = []
        run_args += ["-t", dataset_path]    # input dataset
        run_args += ["-x", "5"]     # x-folds for cross-validation
        run_args += [
            "-classifications",
            "weka.classifiers.evaluation.output.prediction.CSV -file {0} -suppress".format(
                predict_file_path
            )
        ]
        run_args += [
            "-F",
            "weka.filters.unsupervised.attribute.RemoveByName -E ^ID$"
        ]
        run_args += ["-S", "1"]     # Seed
        run_args += ["-W", clf_class]
        if clf_args:
            run_args += ["--"]
            run_args += [str(arg) for arg in copy.copy(clf_args)]

        # Print run info
        self._print_run_info(clf_class, run_args)

        # Add some run args & run classifier
        run_args = [
            "java",
            "-Xmx1024m",
            "-cp",
            self.weka_jar_path,
            "weka.classifiers.meta.FilteredClassifier",
        ] + run_args

        # Run classifier
        output, err, rc = self.run_subprocess(run_args)

        # Save weka outputs, errors and model configuration
        #self._save_stds(stdout_file_path, stderr_file_path,
        #                output, err, rc)
        self._save_model_config(config_file_path, dataset_conf_path,
                                clf_class, clf_args)