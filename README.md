# PV056-AutoML-testing-framework
* *PV056 Machine learning and knowledge discovery*

## How to use PV056-AutoML-testing-framework
* First, follow the [Installation guide](#installation-guide) section
* Then follow the [Usage](#usage) section
 
## Installation guide 
### Prerequisites
- Python version >=3.6 (including python3-dev, python3-wheel)
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/) (comes with the framework)
- Java 8 or later (for Weka execution)
- UNIX-like environment

If you are a Windows user, you can either set up a 
virtual machine or run the script on the Aisa server (the whole script can be run through the terminal, 
even [remotely](https://www.fi.muni.cz/tech/lets-get-started-at-fi.html.cs)).
However, keep in mind that these options will impact the running time of the script.

### Installation to python virtual env (recommended)
It's highly recommended to install this testing framework to python virtual environment.
- Simple python virtual environment guide: https://realpython.com/python-virtual-environments-a-primer/

Simply run commands below in the root folder of this repository.
```
$ python3 -m venv venv
$ source venv/bin/activate # venv/bin/activate.fish # for fish shell
(venv)$ pip install -r requirements.txt
(venv)$ pip install -e .
```

### Installation without python virtual env (not recommended)
```
$ pip install .
```


### Downloading datasets
The framework supports datasets in `.arff` format. The individual datasets can be downloaded from [OpenML](https://www.openml.org).

By default, you should place the downloaded datasets to the `data/datasets/` folder.


## Usage
If you have chosen to install this tester in the virtual environment, you must activate it to proceed.
Virtual environment is activated by running `source venv/bin/activate` and deactivated by `deactivate`


#### Brief commentary on the workflow structure
It is recommended by starting by splitting the data. If you dont want to split the data using this framework,
be aware that you will have to manually create a CSV file that contains tuples of paths to train files and their corresponding test files.

If you dont want to use test files and you dont plan on analyzing differences, use dummy paths.
The framework currently assumes having at least two columns in the input CSVs with paths to datasets,
so if you don't satisfy this criteria when you are doing some part of the workflow manually.

Available workflow steps
 * **SPLIT** (executed once)
 * **FS**, **OD** and **RM** (executed multiple times)
 * **CLF** (executed once with preprocessed data and then optionally with original data)
 * **STATS** (executed once)
 * **GRAPH** (executed multiple times)

**FS** step is not dependent on **OD** and **RM** steps and can be performed before or after those, any number of times.
The **RM** step has to be performed exactly after the **OD** step, as it calculates with a special column that is only
created and kept by the **OD** step. These two steps, **RM** and **OD** are tightly dependent on each other and have to be 
performed together to get any meaningful results, but if you abide by that requirement, they together can be also performed
multiple times independently of **FS** step.

**NOTE!**<br/>*Keep in mind that the OD column and the column with the name ID are checked for by almost every step and is removed to prevent accidental
influence on the performed steps.*

#### Explanation of parameters of commands
Each step has its corresponding command. All are described in detail in the following sections.
They have a lot in common. Each step, except for the **SPLIT**, requires a parameter `-di` (*meaning "datasets input CSV"*) which should contain
a path to a special CSV.

The CSV has to contain a table of at least two columns.
On each row, the first column contains a path to a file intended to train classifiers on and the second column contains a path to a file for testing the same classifier.

The CSV files as described above are created in each step, except for the **STATS** and **GRAPH** steps, and contain paths to modified dataset files, that were created in that step.
User needs to control where the resulting CSV file is created by providing the parameter `-do` (*meaning "datasets output CSV"*).

In addition to input and out CSV paths, a path to a configuration JSON file is required in each step.
The parameter for specifying path to a config JSON is `-c` (*meaning just "config"*).

It should be noted, that the **STATS** step has an additional parameter, through which you can provide baseline classifiers for comparison.

**NOTE!**
<br/>Each command has a builtin `--help` parameter that shows the usage and required parameters.

### Split data
Because we want to cross-validate every classifier, we need to split data into train-test tuples.
Before, this was a part of weka classifiers, yet now, because we want to work with training datasets, we need to do it manually.
An *M\*K-fold cross-validation* is performed based on a configuration JSON. Basically it splits data into *K* folds *M* times.
Resulting into *M\*K* different tuples of train-test files. These tuples will be referred to as splits.

For this purpose, we have the `pv056-split-data`. As specified in its configuration file (see `configs/split/default.json`)
it will split datasets into fifteen train-test tuples and generate a CSV file based on the `-do` parameter,
which is then further used in the workflow.

```
(venv)$ pv056-split-data -h
usage: pv056-split-data [-h] --config-file CONFIG_FILE --datasets-file
                        DATASETS_FILE

Script splits datasets for m*k-fold cross-validation. Script splits datasets
m-times into k folds.

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
  --datasets-file DATASETS_FILE, -do DATASETS_FILE
                        Filename of output datasets csv tracking files
                        relationships
```
#### Example usage
```
(venv)$ pv056-split-data -c configs/split/default.json -do datasets-split.csv
```

#### Example config file
* *data_path*
    * Directory with datasets in arff format
* *train_split_dir*
    * Directory where generated **train** datasets should be saved
* *test_split_dir*
    * Directory where generated **test** datasets should be saved
* *k_of_folds*
    * The number of folds for a generally known k-fold cross-validation.
* *m_of_repeats*
    * Specifies how many different k-fold splits shall be generated.
    * If set to one, data will be split for a basic k-fold cross-validation.
    * Each time a different seed is used, derived deterministically from the random seed.
* *random_state*
    * Optional parameter
    * Specifies a seed for the splitting algorithm.
    * By default it is set to 42. (*See the code*)

```json
{
  "data_path": "data/datasets/",
  "train_split_dir": "data/train_split/",
  "test_split_dir": "data/test_split/",
  "m_of_repeats": 3,
  "k_of_folds": 5
}
```
#### Example output
```
Splitting: weather
Splitting: Weather
Splitting: zoo
Splitting: abalone
Done
```

### Apply outlier detection methods
To apply outlier detection methods to all training splits, we have the `pv056-apply-od-methods`.
This script takes all the training files as described by the first column of the input CSV
and adds a column with outlier detection value. It will generate new training file for every outlier detection method
specified in the configuration JSON file!
```
(venv)$ pv056-apply-od-methods --help
usage: pv056-apply-od-methods [-h] --config-file CONFIG_FILE
                              [-do DATASETS_CSV_OUT] -di DATASETS_CSV_IN

Apply outlier detection methods to training data

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
  -do DATASETS_CSV_OUT, --datasets-csv-out DATASETS_CSV_OUT
                        Path to a csv file which contains result files
                        mappings and their new configurations histories, as
                        modified by this step.
  -di DATASETS_CSV_IN, --datasets-csv-in DATASETS_CSV_IN
                        Path to csv file that contains previous data files
                        mappings, locations and configurations.
```
#### Example usage
```
(venv)$ pv056-apply-od-methods -c configs/od/default.json -di "datasets-split.csv" -do "datasets-od.csv"
```

#### Example config file
* *train_od_dir*
    * Directory where generated **train** datasets with outlier detection values should be saved
* *n_jobs*
    * number of parallel workers
    * You need to have multiple processing units to actually speed up the calculations.
* *times_output*
    * path to file where the run times should be stored.
* *od_methods*
    * List with Outlier detection methods, each methos is described by a two-valued schema.
    * The schema is as follows:
        * *name* - OD name
        * *parameters* - Dictionary "parameter_name": "value"

```json
{
  "train_od_dir": "data/train_od/",
  "n_jobs": 20,
  "times_output": "data/od_times.csv",
  "od_methods": [
    {
      "name": "IsolationForest",
      "parameters": {
        "contamination": "auto",
        "behaviour": "new",
        "random_state": 123
      }
    },
    {
      "name": "LOF",
      "parameters": {
        "contamination": "auto"
      }
    },
    {
      "name": "CODB",
      "parameters": {
        "jar_path": "data/java/WEKA-CODB.jar"
      }
    },
    {
      "name": "ClassLikelihood",
      "parameters": {}
    },
    {
      "name": "NearestNeighbors",
      "parameters": {}
    },
    {
      "name": "ClassLikelihoodDifference",
      "parameters": {}
    },
    {
      "name": "KDN",
      "parameters": {}
    },
    {
      "name": "DS",
      "parameters": {}
    },
    {
      "name": "TD",
      "parameters": {}
    },
    {
      "name": "TDWithPrunning",
      "parameters": {}
    },
    {
      "name": "OneClassSVM",
      "parameters": {
        "gamma": "scale"
      }
    }
  ]
}
```
#### Example output
```
Working on  IsolationForest: weather_2-1_90a...ain.arff --> weather_2-1_90a...a880a9ff_train.arff
Working on  IsolationForest: weather_3-0_909...ain.arff --> weather_3-0_90a...a880a9ff_train.arff
Working on  IsolationForest: weather_3-2_90a...ain.arff --> weather_3-2_90a...a880a9ff_train.arff
Working on  IsolationForest: weather_2-0_90a...ain.arff --> weather_2-0_90a...a880a9ff_train.arff
.
.
.
One of the workers reached an empty queue
One of the workers reached an empty queue
One of the workers reached an empty queue
Done
```


#### Outlier detector names and parameters:
| Name | Full name | Parameters |
|:----:|:----------:|:----------:|
| **LOF** | Local Outlier Factor | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) |
| **NearestNeighbors** | Nearest Neighbors | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) |
| **IsolationForest** | Isolation Forest | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
| **OneClassSVM** | One-class SVM | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) |
| **EllipticEnvelope** | Elliptic Envelope | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html) |
| **ClassLikelihood** | Class Likelihood | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html)
| **ClassLikelihoodDifference** | Class Likelihood Difference | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html)
| **KDN** | K-Disagreeing Neighbors | n_neighbors |
| **DS** | Disjunct size | -- |
| **DCP** | Disjunct class percentage | min_impurity_split [docs](https://blog.nelsonliu.me/2016/08/05/gsoc-week-10-scikit-learn-pr-6954-adding-pre-pruning-to-decisiontrees/) |
| **TD** | Tree Depth with and without prunning | -- |
| **TDWithPrunning** | Tree Depth with prunning | min_impurity_split |
| **Random** | Random OD score | seed |
| **CODB** | CODB | See below |

**NOTE!**<br/>
The DCP detector currently causes the processes to hang, so it is advise to use it carefully and check if all files are really created.
The main process after completion has to be terminated manually.


#### CODB
* path to CODB jar file jar_path, must be defined
* k nearest neighbors (default = 7) -k "\<int\>"
* Alpha coeffecient (default = 100) -a "\<double\>"
* Beta coeffecient (default = 0.1) -b "\<double\>"
* distance-type (default = motaz.util.EuclidianDataObject) -D "\<String\>"
* Replace Missing Vaules (default = false) -r 
* Remove Missing Vaules (default = false) -m

Example:
```
{
    "train_split_dir": "data/train_split/",
    "train_od_dir": "data/train_od/",
    "n_jobs": 2,
    "times_output": "outputs/od_times.csv",
    "od_methods": [
        {
            "name": "CODB",
            "parameters": {
                "jar_path" : "data/java/WEKA-CODB.jar",
                "-k" : "10",
                "-r" : "",
                "-m": ""
            }
        }
    ]
}
```

* New methods for outlier detection coming sooner or later, probably later!


### Remove outliers
As the name suggests, this script will remove the biggest outliers from each training split with outlier detection values.
It is absolutely necessary for this step that the dataset files contain a column with outlier detection score values.
These score values are generated by the **OD** step and are discarded by every other step to avoid interference with classification or feature selection.

```
(venv)$ pv056-remove-outliers  --help
usage: pv056-remove-outliers [-h] --config-file CONFIG_FILE --datasets-csv-in
                             DATASETS_CSV_IN --datasets-csv-out
                             DATASETS_CSV_OUT

Removes the percentage of the largest outliers.

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
  --datasets-csv-in DATASETS_CSV_IN, -di DATASETS_CSV_IN
                        CSV file with input dataset split mappings and
                        configuration histories
  --datasets-csv-out DATASETS_CSV_OUT, -do DATASETS_CSV_OUT
                        CSV containing resulting dataset split mappings and
                        updated configuration histories
```
#### Example usage
```
(venv)$ pv056-remove-outliers  -c configs/rm/default.json -di "datasets-od.csv" -do "datasets-rm.csv"
```

#### Example config file
* *train_od_dir*
    * generated **train** datasets with outlier detection values
* *train_removed_dir*
    * Directory where train data with **removed** outliers should be saved
* *percentage*
    * How many percents of the largest outliers should be removed (0.0-100.0)
    * If you want to remove the observations that are least anomalous instead, use negative values.
    * int or List[int]
```json
{
    "train_od_dir": "data/train_od/",
    "train_removed_dir": "data/train_removed/",
    "keep_original": true,
    "percentage": [ 1, 2, 5, -1, -2, -5 ]
}
```
#### Example output
```
Removing [0.5, 2.0, 5.0, -0.5, -2.0, -5.0]%
    data/train_od/weather_2-1_90a29a4d22830e7af8d582b68775e47c_FSf618edfe86acc6636049013b543b8a86_OD-414bd82745e9a87176dd4401a880a9ff_train.arff 0.5%
    data/train_od/weather_2-1_90a29a4d22830e7af8d582b68775e47c_FSf618edfe86acc6636049013b543b8a86_OD-414bd82745e9a87176dd4401a880a9ff_train.arff 2.0%
    data/train_od/weather_2-1_90a29a4d22830e7af8d582b68775e47c_FSf618edfe86acc6636049013b543b8a86_OD-414bd82745e9a87176dd4401a880a9ff_train.arff 5.0%
    .
    .
    .
    data/train_od/abalone_0-0_90a29a4d22830e7af8d582b68775e47c_FS2e4d042a1485be0d036bb56351342a4c_OD-9b61d27acb971cb642c93a0a0c4f2ed7_train.arff -0.5%
    data/train_od/abalone_0-0_90a29a4d22830e7af8d582b68775e47c_FS2e4d042a1485be0d036bb56351342a4c_OD-9b61d27acb971cb642c93a0a0c4f2ed7_train.arff -2.0%
    data/train_od/abalone_0-0_90a29a4d22830e7af8d582b68775e47c_FS2e4d042a1485be0d036bb56351342a4c_OD-9b61d27acb971cb642c93a0a0c4f2ed7_train.arff -5.0%
Done

```


### Perform feature selection (FS step)
At this point, the feature selection step allows you to run various WEKA or SCIKIT methods for feature selection.
Current implementation allows you to specify a library, its method and parameters.

```
(venv)$ pv056-evaluate-features --help
usage: pv056-evaluate-features [-h] -c CONFIG_FS [-do DATASETS_CSV_OUT] -di
                               DATASETS_CSV_IN

PV056-AutoML-testing-framework

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FS, --config-fs CONFIG_FS
                        path to feature selection config file
  -do DATASETS_CSV_OUT, --datasets-csv-out DATASETS_CSV_OUT
                        path to a csv file which contains datasets used for FS
                        and their respective result files
  -di DATASETS_CSV_IN, --datasets-csv-in DATASETS_CSV_IN
                        Path to csv file that contains previous data files
                        mappings, locations and for example OD configurations
```
#### Example usage
```
(venv)$ pv056-evaluate-features -c configs/fs/default.json -di "datasets-rm.csv" -do "datasets-fs.csv"
```

#### Example of a config file
* *output_folder_path*
    * The directory where the result files are stored.
* *weka_jar_path*
    * Path to a WEKA jar file
* *blacklist_file_path*
    * Path to a file containing blacklisted combinations of dataset and FS method
* *selection_methods*
    * List[] of of feature selection method configurations.
    * Each FS method config in the list has to follow a schema. The schema contains parameter
    *source_library* which specifies which library to pick selection method from.
     Currently two libraries are available: `WEKA` and `SCIKIT`
    * In case of feature selection, the implementation usually requires two methods.
    One method usually divides features into special subsets or creates rankings,
     the other method selects which features to actually filter out.
     The parameter names `fs_method`, `score_func` in WEKA and `eval_class`, `search_class` in WEKA
     follow the naming and structure of the methods in their respective libraries.
     It is advised to look into the documentation of said libraries to gain basic understanding of how to use them.
     To make it easier, here are links to [SCIKIT feature selection package documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)
     and to the [WEKA documentation index](https://weka.sourceforge.io/doc.stable-3-8/) where you can search for specific method details.
     It is also recommended to install and explore the WEKA GUI to gain basic understanding of how the methods interact
     and which parameters they use.
    * *leave_attributes_binarized*
        * This is a special parameter specific for SCIKIT only. You will probably not need to use it,
        but in case you have very large numbers of categorical attributes with very large domains,
        setting this attribute to false might help reduce the calculation times of subsequent steps.
        Before using this, read the implementation details and how it works in my [thesis](https://is.muni.cz/th/eh6aj/) in the chapter 4.3.2 (page 15)
        


```
{
  "output_folder_path": "data/fs_outputs/",
  "weka_jar_path": "data/java/weka.jar",
  "blacklist_file_path": "fs_blacklist.csv",
  "selection_methods": [
    {
      "source_library": "SCIKIT",
      "leave_attributes_binarized": true,
      "fs_method": {
        "name": "SelectKBest",
        "parameters": {
          "k": 4
        }
      },
      "score_func": {
        "name": "f_classif",
        "parameters": {}
      }
    },
    {
      "source_library": "WEKA",
      "eval_class": {
        "name": "weka.attributeSelection.InfoGainAttributeEval",
        "parameters": {}
      },
      "search_class": {
        "name": "weka.attributeSelection.Ranker",
        "parameters": {
          "T": 5E-3,
          "N": "50%"
        }
      }
    }
  ]
}
```
#### Example output
```
Starting feature selection step
Queue filled! Initializing individual workers for parallel processing...
Processing FS for f_classif --> weather .
Binarizing test data for f_classif weather
Processing FS for f_classif --> weather .
.
.
.
Completed FS for  f_classif on file abalone .
Processing FS for weka.attributeSelection.InfoGainAttributeEval --> abalone .
Completed FS for  weka.attributeSelection.InfoGainAttributeEval on file abalone .
Completed FS for  f_classif on file abalone .
Processing FS for weka.attributeSelection.InfoGainAttributeEval --> abalone .
Completed FS for  weka.attributeSelection.InfoGainAttributeEval on file abalone .
Done

```


### Run weka classifiers (CLF step)
As in other steps, to run this step, you need to have an input CSV with train and test filepaths columns
and optionally additional columns with paths to JSON files with configurations of preprocessing steps applied to the train files.
As usual, the path to the input CSV is specified via the  `--datasets-csv-in` or `-di` parameter.
The other parameter, `--datasets-csv-out` or `-do` will specify a filepath to the output CSV.
The output CSV here will contain paths to files with predictions of the trained classifiers and paths to JSON files
with complete list of configurations of used preprocessing steps. 

```
(venv) aura:/var/tmp/AutoMLref>$ pv056-run-clf --help
usage: pv056-run-clf [-h] -c CONFIG_CLF -di DATASETS_CSV_IN -do
                     DATASETS_CSV_OUT

PV056-AutoML-testing-framework

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_CLF, --config-clf CONFIG_CLF
                        path to classifiers config file
  -di DATASETS_CSV_IN, --datasets-csv-in DATASETS_CSV_IN
                        Path to csv with train/test/configs filepaths.
  -do DATASETS_CSV_OUT, --datasets-csv-out DATASETS_CSV_OUT
                        Path to csv with predictions, test files, configs
                        filepaths.
```

#### Example usage
```
(venv)$ pv056-run-clf -c configs/clf/default.json -di "datasets-fs.csv" -do "datasets-clf.csv"
```

#### Example of config file for weka classifiers
* *output_folder*
    * path to output folder, where outputs from your classifiers will be saved
* *weka_jar_path*
    * path to a weka.jar file
* *n_jobs*
    * number of parallel workers
* *times_output*
    * path to file where the run times should be stored
* *timeout*
    * max run time before a classifier is terminated (in seconds)
* *classifiers*
    * list of classifiers which you want to run
    * you can run an arbitrary number of classifiers, even same classifier with different configuration
    * list of [weka classifiers](http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html)
    * *class_name*
        * name of weka classifier class
    * *args*
        * optional value
        * list of arguments for specific classifier
        * you can find all arguments for specific classifier using weka command: ```$ java -cp weka.jar weka.classifiers.trees.J48 --help``` or in [weka documentation](http://weka.sourceforge.net/doc.dev/weka/classifiers/trees/J48.html)
        * you can find more information about Weka-CLI in the section below [How to work with Weka 3](#how-to-work-with-Weka-3), but I don't think you need that for using this tool.
    * *filters*
        * optional value
        * you can use any filter from Weka
        * you have to specify name of filter and arguments for it

```json
{
    "output_folder": "clf_outputs/",
    "weka_jar_path": "data/java/weka.jar",
    "n_jobs": 20,
    "times_output": "outputs/clf_times.csv",
    "timeout": 1800,
    "classifiers": [
        {
            "class_name": "weka.classifiers.trees.J48",
            "args": [
                "-C",
                0.25,
                "-M",
                2
            ],
            "filters": [
                {
                    "name": "weka.filters.unsupervised.attribute.RemoveByName",
                    "args": [
                        "-E",
                        "^size$"
                    ]
                }
            ]
        },
        {
            "class_name": "weka.classifiers.trees.J48",
            "args": [
                "-C",
                0.35
            ]
        },
        {
            "class_name": "weka.classifiers.bayes.BayesNet"
        }
    ]
}
```
#### Example output
```
Starting CLF step...
Filling up the jobs queue for parallel processing...
25902 of total jobs will be generated for 8634 files and 3 classifiers.
Queue filled up!
Starting individual workers...
Starting job number 8.
8. job done.
Starting job number 1.
1. job done.
Starting job number 0.
0. job done.
Starting job number 3.
3. job done.
.
.
.

```

### Count accuracy
To count accuracy simply run `pv056-statistics` script. Script will put it all together
and generate output in csv format (see example below).
What you need to specify is the path to a results CSV from a **CLF** step as a `--datasets-csv-in` or `-di`
parameter, and optional baseline results CSV, from a different **CLF** step, to compare it to.
The parameter to specify a path to the baseline CSV is `--datasets-csv-baseline` or `-db`.

**NOTE!**<br/>
Right now the intended use is that the baseline **CLF** step is performed on the unprocessed train splits.
It was not tested for comparing any baseline. For details look into the code. It might work, but it is not guaranteed.
```
(venv)$ pv056-statistics --help
usage: pv056-statistics [-h] --config-file CONFIG_FILE --datasets-csv-in
                        DATASETS_CSV_IN
                        [--datasets-csv-baseline DATASETS_CSV_BASELINE]

Script for counting basic statistic (Accuracy, )

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
  --datasets-csv-in DATASETS_CSV_IN, -di DATASETS_CSV_IN
                        CSV file with paths to predictions, test files and
                        configurations histories
  --datasets-csv-baseline DATASETS_CSV_BASELINE, -db DATASETS_CSV_BASELINE
                        CSV file with paths to predictions, test files and
                        configurations histories
```
#### Example usage
```
pv056-statistics -c configs/stats/default.json -di datasets-clf.csv -db -datasets-base.csv
```

#### Example of config file for statistics
* *output_table*
    * path to a CSV file where the completed statistics will be saved.
* *aggregate*
    * whether to aggregate values across folds (using *mean*)
* *pattern(removed)* 
    * regex pattern which of the files should be taken into account
    (e.g. when you are only interested in the results on some datasets)
    * this feature was removed during the changes I have made.
    The files are now not loaded from a directory, but from a CSV file
     and the CSV can be easily modified to obtain the same functionality.
    * The feature can be added back, if it would be demanded by users.
    
```JSON
{
  "output_table": "outputs/results.csv",
  "aggregate": true,
  "pattern": ".*"
}
```

#### Example output
```
(venv)$ cat outputs/results.csv
dataset,clf,clf_family,clf_params,step_0,step_1,step_2,step_3,accuracy,accuracy_base,gain
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': -5.0},"{'eval_class': {'name': 'weka.attributeSelection.CfsSubsetEval', 'parameters': {'E': 4, 'L': True, 'M': True, 'P': 3, 'Z': True}}, 'search_class': {'name': 'weka.attributeSelection.BestFirst', 'parameters': {'N': 5, 'S': 3}}, 'source_library': 'WEKA'}",0.79621,0.79691,0.0007000000000000339
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': -5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': False, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.77951,0.79691,0.01739999999999997
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': -5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': True, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.80512,0.79691,-0.00820999999999994
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': 5.0},"{'eval_class': {'name': 'weka.attributeSelection.CfsSubsetEval', 'parameters': {'E': 4, 'L': True, 'M': True, 'P': 3, 'Z': True}}, 'search_class': {'name': 'weka.attributeSelection.BestFirst', 'parameters': {'N': 5, 'S': 3}}, 'source_library': 'WEKA'}",0.80679,0.79691,-0.00988
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': 5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': False, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.77728,0.79691,0.019630000000000036
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': 5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': True, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.81347,0.79691,-0.01656000000000002
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': -5.0},"{'eval_class': {'name': 'weka.attributeSelection.CfsSubsetEval', 'parameters': {'E': 4, 'L': True, 'M': True, 'P': 3, 'Z': True}}, 'search_class': {'name': 'weka.attributeSelection.BestFirst', 'parameters': {'N': 5, 'S': 3}}, 'source_library': 'WEKA'}",0.79621,0.79691,0.0007000000000000339
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': -5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': False, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.77951,0.79691,0.01739999999999997
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': -5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': True, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.80512,0.79691,-0.00820999999999994
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': 5.0},"{'eval_class': {'name': 'weka.attributeSelection.CfsSubsetEval', 'parameters': {'E': 4, 'L': True, 'M': True, 'P': 3, 'Z': True}}, 'search_class': {'name': 'weka.attributeSelection.BestFirst', 'parameters': {'N': 5, 'S': 3}}, 'source_library': 'WEKA'}",0.80345,0.79691,-0.00653999999999999
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': 5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': False, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.78229,0.79691,0.014619999999999966
anneal,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': 5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': True, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.81793,0.79691,-0.02102000000000004
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': -5.0},"{'eval_class': {'name': 'weka.attributeSelection.CfsSubsetEval', 'parameters': {'E': 4, 'L': True, 'M': True, 'P': 3, 'Z': True}}, 'search_class': {'name': 'weka.attributeSelection.BestFirst', 'parameters': {'N': 5, 'S': 3}}, 'source_library': 'WEKA'}",0.62168,0.62629,0.004610000000000003
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': -5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': False, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.65044,0.62629,-0.024150000000000005
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': -5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': True, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.62832,0.62629,-0.0020299999999999763
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': 5.0},"{'eval_class': {'name': 'weka.attributeSelection.CfsSubsetEval', 'parameters': {'E': 4, 'L': True, 'M': True, 'P': 3, 'Z': True}}, 'search_class': {'name': 'weka.attributeSelection.BestFirst', 'parameters': {'N': 5, 'S': 3}}, 'source_library': 'WEKA'}",0.60619,0.62629,0.020100000000000007
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': 5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': False, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.61947,0.62629,0.006820000000000048
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'CODB', 'parameters': {'jar_path': 'data/java/WEKA-CODB.jar'}}",{'RM': 5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': True, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.61947,0.62629,0.006820000000000048
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': -5.0},"{'eval_class': {'name': 'weka.attributeSelection.CfsSubsetEval', 'parameters': {'E': 4, 'L': True, 'M': True, 'P': 3, 'Z': True}}, 'search_class': {'name': 'weka.attributeSelection.BestFirst', 'parameters': {'N': 5, 'S': 3}}, 'source_library': 'WEKA'}",0.62168,0.62629,0.004610000000000003
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': -5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': False, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.65044,0.62629,-0.024150000000000005
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': -5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': True, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.62832,0.62629,-0.0020299999999999763
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': 5.0},"{'eval_class': {'name': 'weka.attributeSelection.CfsSubsetEval', 'parameters': {'E': 4, 'L': True, 'M': True, 'P': 3, 'Z': True}}, 'search_class': {'name': 'weka.attributeSelection.BestFirst', 'parameters': {'N': 5, 'S': 3}}, 'source_library': 'WEKA'}",0.62168,0.62629,0.004610000000000003
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': 5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': False, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.62389,0.62629,0.0024000000000000687
audiology,DecisionTable,rules,[],"{'data_path': 'data/datasets/', 'k_of_folds': 2, 'm_of_repeats': 2, 'random_state': 42, 'test_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/test_split/', 'train_split_dir': '/var/tmp/xbajger/BAKPR/data_exp/train_split/'}","{'name': 'IsolationForest', 'parameters': {'behaviour': 'new', 'contamination': 'auto', 'random_state': 123}}",{'RM': 5.0},"{'fs_method': {'name': 'SelectFpr', 'parameters': {'alpha': 0.2}}, 'leave_attributes_binarized': True, 'score_func': {'name': 'chi2', 'parameters': {}}, 'source_library': 'SCIKIT'}",0.62389,0.62629,0.0024000000000000687

```

### Generate graphs
To gain detailed understanding of this feature, refer to the chapter 5.3.6 (page 36) of my [bachelor thesis](https://is.muni.cz/auth/th/eh6aj/).
You need to specify graph configuration to determine the layout and visualisation targets, input data (result CSV from a workflow) and destination for the final PNG.
```
(venv) aura:/var/tmp/AutoMLref>$ pv056-graph-scatter --help
Starting graph creation, parsing arguments.
usage: pv056-graph-scatter [-h] -c CONFIG_GRAPH [-o OUTPUT_PNG] -di
                           DATASETS_CSV_IN

PV056-AutoML-testing-framework

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_GRAPH, --config-graph CONFIG_GRAPH
                        path to visualisation config file
  -o OUTPUT_PNG, --output-png OUTPUT_PNG
                        path to a png file which contains visualized results
                        in a nice graph
  -di DATASETS_CSV_IN, --datasets-csv-in DATASETS_CSV_IN
                        Path to csv file that contains results from a workflow
```
```
(venv) aura:/var/tmp/AutoMLref>$ pv056-graph-box --help
Starting to create boxplot, parsing arguments...
usage: pv056-graph-box [-h] -c CONFIG_GRAPH [-o OUTPUT_PNG] -di
                       DATASETS_CSV_IN

PV056-AutoML-testing-framework

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_GRAPH, --config-graph CONFIG_GRAPH
                        path to visualisation config file
  -o OUTPUT_PNG, --output-png OUTPUT_PNG
                        path to a png file which contains visualized results
                        in a nice graph
  -di DATASETS_CSV_IN, --datasets-csv-in DATASETS_CSV_IN
                        Path to csv file that contains results from a workflow
```
#### Example usage
```
(venv)$ pv056-graph-scatter -c configs/graph/default_scatter.json -di outputs/results.csv -o my_scatterplot.png
```
```
(venv)$ pv056-graph-box -c configs/graph/default_box.json -di outputs/results.csv -o my_boxplot.png
```

#### Example configurations
Scatterplot configuration:
```
{
  "col_examined": "gain",
  "col_related": "step_1",
  "col_grouped_by": ["step_2", "step_3"],
  "legend_title": "Legend title specified by user\nallows linebreaks",
  "title": "Evaluation of accuracy gain across multiple configurations",
  "x_title": "Title for X axis",
  "y_title": "Accuracy gain",
  "convert_col_related_from_json": true,
  "dpi": 200,
  "width_multiplier": 1.1,
  "height_multiplier": 1
}
```
Boxplot configuration:
```
{
  "sort_by_column": "accuracy",
  "col_examined": "gain",
  "col_related": "step_1",
  "title": "Main title desribing the graph",
  "x_title": "Configurations of step 1 (you must know which one it was)",
  "y_title": "Gain in accuracy",
  "sort_func_name": "inv_mean",
  "extract_col_related": "name",
  "show_fliers": true,
  "dpi": 300,
  "width_multiplier": 0.8,
  "height_multiplier": 1
}
```


## All-in-one script
To make the script easier to use, we have created a runnable shell script which runs the full
OD pipeline without the need to execute each step separately.

The script runs each step with its respective default configuration file found in the ```configs/```
folder and its subdirectories. Assuming the virtual environment is set up (see [Installation guide](#installation-guide)), all you need to do is set the config files according to your
preference and then run the script from the root directory of the project.

The outputs for each step can be found in the ```logs/``` directory.

***Note:** there is no need to activate the virtual environment first as the the
script takes care of it, as well as deactivating it after the script finishes*

#### Example usage
```
>$ sh scripts/script.sh
05/12/2019-17:21:33 - Splitting data...
05/12/2019-17:21:35 - SPLIT done.

05/12/2019-17:21:35 - Applying OD methods...
05/12/2019-17:21:36 - OD done.

05/12/2019-17:21:36 - Removing outliers...
05/12/2019-17:21:38 - RM done.

05/12/2019-17:21:38 - Running classification...
05/12/2019-17:21:57 - CLF done.

05/12/2019-17:21:57 - Counting accuracy...
05/12/2019-17:21:58 - ACC done.

Script finished.

```

If you want to execute the scripts on remote servers and you are new to unix, it might be handy to look
into the `disown` command and `&` operator. With those you can execute the command to run in background even if the session
is terminated. You also need to redirect all outputs, so you have it at hand when the script finishes.
You will also probably want to ot take up all the server resources, sou might as well want to use the `nice` command (for example if you are on faculty servers)
You can do it as follows:

```
$ nice -n 14 sh scripts/script.sh > file_for_outputs.log 2>&1 & disown
```
the above command will execute your workflow (as programmed in the `scripts/script.sh` file), redirects stdout to the `file_for_outputs.log`
file and also redirects the stderr to the stdout, so you should have all the eventual warnings and errors, or just logs, in one file.
### Checking progress
Depending on your setup, the script may take a long time to finish. In order to check how far the script got,
you should peek into the `od_times.csv` and `clf_times.csv` files generated during execution for the progress
on outlier detection and classification, respectively.
These files list the datasets already processed with the time it took for each of them.
Also, some logging was added as a part of the standard output during the execution, so you should be able to see
how many jobs were completed by the workers already in some parallel steps, such as FS, or CLF.

## Reporting problems
All encountered problems regarding running the script should be posted to the subject [forum](https://is.muni.cz/auth/discussion/predmetove/fi/jaro2020/PV056/). This is mainly
because if someone struggles with something, chances are other students have encountered the same problem.

## How to work with Weka 3
* Download Weka from https://www.cs.waikato.ac.nz/ml/weka/downloading.html
* Weka classifiers http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html
* Documentation to Weka in `The WEKA Workbench` & `Weka manual` documents at https://www.cs.waikato.ac.nz/ml/weka/documentation.html
```shell
# Example of running J48 classifier with diabetes dataset
java -cp weka.jar weka.classifiers.trees.J48 -t data/diabetes.arff

# General & Specific configuration options
java -cp weka.jar weka.classifiers.trees.J48 --help
```

## Developers guide

As developers, we have chosen to use [Black](https://github.com/ambv/black/) auto-formater and [Flake8](https://gitlab.com/pycqa/flake8) style checker. Both of these tools are pre-prepared for pre-commit. It's also recommended to use [mypy](https://github.com/python/mypy).


Since there is the typing module in the standard Python library, it would be a shame not to use it.  A wise old man once said: More typing, fewer bugs. [Typing module](https://docs.python.org/3/library/typing.html)


To prepare dev env run commands below.
```
$ python3 -m venv venv
$ source venv/bin/activate # venv/bin/activate.fish # for fish shell
(venv)$ pip install -r requirements.txt
(venv)$ pip install -r requirements-dev.txt
(venv)$ pre-commit install
(venv)$ pip install -e .
```

For generating `requirements.txt` we are using pip-compile from [pip-tools](https://github.com/jazzband/pip-tools).
For keeping your packages updated, use `pip-sync requirements.txt requirements-dev.txt`.
