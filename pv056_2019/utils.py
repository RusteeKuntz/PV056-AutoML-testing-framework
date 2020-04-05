# *********************************************************
# Utils for classifiers
# *********************************************************

ID_NAME = "ID"
OD_VALUE_NAME = "OD_VALUE"

# *********************************************************
# Other utils
# *********************************************************

BASELINE_NAME = "baseline"


def valid_path(path, message):
    import os
    import argparse
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(message)
    else:
        return path
