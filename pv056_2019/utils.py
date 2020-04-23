# *********************************************************
# Utils for classifiers
# *********************************************************

ID_NAME = "ID"
OD_VALUE_NAME = "OD_VALUE"

# *********************************************************
# Other utils
# *********************************************************

BASELINE_NAME = "baseline"
NONE_STR = "none"

# library identifiers
CUSTOM = "CUSTOM"
WEKA = "WEKA"
SCIKIT = "SCIKIT"


def valid_path(path, message):
    import os
    import argparse
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(message)
    else:
        return path


# df = pd.DataFrame([[1, 2, 2, 3, 4], [2, 1, 0, 0, 4], [0, 4, 15, 20, 9], [2, 3, 3, 11, 20]], columns=["a", "b", "c", "d", "e"])

UNSORTED_DATASETS = [
    [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json']]
DATASETS = [
    [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/ar1_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/ar1_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/anneal_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-2b3b3b9968ee9ed197d758dfcdd97cdf_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/anneal_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/2b3b3b9968ee9ed197d758dfcdd97cdf.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_1-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_1-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM1.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM1.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-1_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-1_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json'], [
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/audiology_0-0_OD-414bd82745e9a87176dd4401a880a9ff_RM-5.00_FS-438da84bb8ba84e0a37ebc9adcfa49d3_train.arff',
        '/var/tmp/xbajger/BAKPR/data_exp/test_split/audiology_0-0_test.arff', '',
        '/var/tmp/xbajger/BAKPR/data_exp/train_od/414bd82745e9a87176dd4401a880a9ff.json',
        '/var/tmp/xbajger/BAKPR/data_exp/train_removed/RM-5.00.json',
        '/var/tmp/xbajger/BAKPR/data_exp/fs_outputs/438da84bb8ba84e0a37ebc9adcfa49d3.json']]
