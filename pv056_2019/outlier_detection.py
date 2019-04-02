from __future__ import absolute_import

import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors, KNeighborsClassifier
from .F2 import F2Metric
from .T1 import T1Metric
from .MV import MVMetric
from .CB import CBMetric
from TD import TDMetric
from DCP import DCPMetric
from DS import DSMetric
from KDN import KDNMetric


DETECTORS: Dict[str, Any] = {}


class AbstractDetector:
    name: str
    data_type: str
    values: np.array

    def __init__(self, **settings):
        self.settings = settings

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        raise NotImplementedError()


def detector(cls):
    DETECTORS.update({cls.name: cls})
    return cls


@detector
class LOF(AbstractDetector):
    name = "LOF"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = LocalOutlierFactor(**self.settings)
        self.clf.fit(bin_dataframe.values)
        self.values = self.clf._decision_function(bin_dataframe.values)
        return self


@detector
class NN(AbstractDetector):
    name = "NearestNeighbors"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        if "n_neighbors" in self.settings:
            self.settings["n_neighbors"] = int(self.settings["n_neighbors"])
        self.clf = NearestNeighbors(**self.settings)
        self.clf.fit(bin_dataframe.values)
        distances, _ = self.clf.kneighbors()
        self.values = np.mean(distances, axis=1)
        return self


@detector
# k-Disagreeing neighbors: The percentage of the
# k nearest neighbors (using Euclidean
# distance) for an instance that do not share its target class value
# TODO possibility of adding k into config file


class KDN(AbstractDetector):
    name = "KDN"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        if "n_neighbors" in self.settings:
            k = int(self.settings["n_neighbors"])
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = KDNMetric()
        self.values = self.clf.countKDN(bin_dataframe, classes, k)
        print("KDN done sucessfully!")
        return self


@detector
# Disjunct size: The number of instances covered by a disjunct that the investigated instance
# belongs to divided by the number of instances covered by the largest disjunct in an
# unpruned decision tree
class DS(AbstractDetector):
    name = "DS"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = DSMetric()
        self.values = self.clf.countDS(bin_dataframe, classes)
        print("DS done sucessfully!")
        return self


@detector
# Disjunct class percentage: The number of instances in a disjunct that have the same class
# label as the investigated instance divided by the total number of instances in the disjunct in a
# pruned decision tree
class DCP(AbstractDetector):
    name = "DCP"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        if "min_impurity_split" in self.settings:
            minimum_impurity_split = float(self.settings["min_impurity_split"])
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = DCPMetric()
        self.values = self.clf.countDCP(bin_dataframe, classes, minimum_impurity_split)
        print("DCP done sucessfully!")
        return self


@detector
# Tree depth: The depth of the leaf node that classifies an instance in an induced decision tree without prunning
class TD(AbstractDetector):
    name = "TD"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = TDMetric()
        self.values = self.clf.findLeafDepthWithoutPrunning(bin_dataframe, classes)
        print("TD without prunning done sucessfully!")
        return self


@detector
# Tree depth: The depth of the leaf node that classifies an instance in an induced decision tree with prunning
class TDWithPrunning(AbstractDetector):
    name = "TDWithPrunning"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        if "min_impurity_split" in self.settings:
            minimum_impurity_split = float(self.settings["min_impurity_split"])
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = TDMetric()
        self.values = self.clf.findLeafDepthWithPrunning(
            bin_dataframe, classes, minimum_impurity_split
        )
        print("TD with prunning done sucessfully!")
        return self


# @detector
# # Error rate of 1NN classifier: Leave-one-out error estimate of 1NN
# class N3(AbstractDetector):
#     name = "N3"
#     data_type = "REAL"

#     def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
#         self.sum = 0
#         self.clf = NearestNeighbors(n_neighbors=1)
#         classColumnIndex = len(dataframe.columns) - 1
#         classColumnName = dataframe.columns[classColumnIndex]
#         neigh = KNeighborsClassifier(n_neighbors=1)
#         bin_dataframe = dataframe._binarize_categorical_values()
#         for index, row in dataframe.iterrows():
#             print("Training " + repr(index) + ". classifier.")
#             leaveOne = dataframe.index.isin([index])
#             dataframeMinusOne = (dataframe[~leaveOne]).reset_index(drop=True)
#             bin_dataframeMinusOne = bin_dataframe[~leaveOne].reset_index(drop=True)
#             neigh.fit(bin_dataframeMinusOne, dataframeMinusOne[classColumnName])
#             indices = neigh.kneighbors(
#                 bin_dataframe[leaveOne], n_neighbors=1, return_distance=False
#             )
#             for i in indices:
#                 if (
#                     dataframeMinusOne[classColumnName][i[0]]
#                     != dataframe[classColumnName][index]
#                 ):
#                     self.sum += 1
#         self.values = self.sum / len(dataframe)
#         print("N3 done sucessfully!")
#         return self


@detector
class IsoForest(AbstractDetector):
    name = "IsolationForest"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = IsolationForest(**self.settings)
        self.clf.fit(bin_dataframe.values)
        self.values = self.clf.decision_function(bin_dataframe.values)
        return self


@detector
class F2(AbstractDetector):
    name = "F2"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = F2Metric()
        self.values = self.clf.compute_values(df=bin_dataframe, classes=classes)
        return self


@detector
class T1(AbstractDetector):
    name = "T1"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = T1Metric()
        self.values = self.clf.compute_values(df=bin_dataframe)
        return self


@detector
class T2(AbstractDetector):
    name = "T2"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        samples_count = len(bin_dataframe.index)
        features_count = len(bin_dataframe.columns)
        self.values = samples_count / features_count
        return self


@detector
class MV(AbstractDetector):
    name = "MV"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):

        self.clf = MVMetric()
        self.values = self.clf.compute_values(classes=classes)
        return self


@detector
class CB(AbstractDetector):
    name = "CB"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):

        self.clf = CBMetric()
        self.values = self.clf.compute_values(classes=classes)
        return self
