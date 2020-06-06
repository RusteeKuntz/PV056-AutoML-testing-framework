from __future__ import absolute_import

import os
import warnings
import re
import resource
from typing import Any, Dict, List, Optional

import arff
import numpy as np
import pandas as pd
import time

from sklearn.feature_selection.univariate_selection import _BaseFilter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from pv056_2019.feature_selection import F_SELECTORS, AbstractFeatureSelector
from pv056_2019.outlier_detection import DETECTORS
from pv056_2019.utils import ID_NAME, OD_VALUE_NAME, WEKA_DATA_TYPES, ArffData
from pv056_2019.schemas import OutlierDetectorSchema, CustomFSSchema

warnings.simplefilter(action="ignore", category=UserWarning)


class DataFrameArff(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        print("INIT DAtaFrameArff")
        arff_data: Optional[dict or ArffData] = kwargs.pop("arff_data", None)
        if isinstance(arff_data, ArffData):
            arff_data = arff_data.__dict__
        if arff_data is None:
            super().__init__(*args, **kwargs)
        else:
            columns = [x[0] for x in arff_data["attributes"]]

            super().__init__(arff_data["data"], columns=columns, **kwargs)

            self._arff_data: Dict[str, Any] = {}
            for key, item in arff_data.items():
                if key.lower() != "data":
                    self._arff_data.update({key: item})
            self._arff_data["relation"] = self._arff_data["relation"].replace("_", "-")
        print("INIT DONE")

    def arff_data(self) -> ArffData:
        data = self._arff_data
        data.update({"data": self.replace(np.nan, None).values.tolist()})
        return ArffData(**data)

    def arff_dumps(self):
        return arff.dumps(self.arff_data().__dict__)

    def arff_dump(self, file_path: str):
        with open(file_path, "w") as output_file:
            data = self._arff_data
            data.update({"data": self.replace(np.nan, None).values.tolist()})
            arff.dump(data, output_file)

    # this encodes nominal values through one-hot encoding.
    # Also (to my surprise - xbajger) it removes the last attribute (usually the class, the framework depends on this)
    # class attribute is removed before binarization, so that is is not binarized.
    def _binarize_categorical_values(self) -> pd.DataFrame:
        encoded_dataframe = pd.DataFrame()
        for attr, values in self._arff_data["attributes"][:-1]:
            enc = OneHotEncoder(handle_unknown="ignore")
            enc.fit(np.array(values).reshape(-1, 1))
            if isinstance(values, list):
                transformed_data = enc.transform(
                    self[attr].values.reshape(-1, 1)
                ).toarray()
                columns_index = pd.MultiIndex.from_product(
                    [[attr], values], names=["0", "1"]
                )
            elif values.lower() in {"numeric", "real", "integer"}:
                imputer = SimpleImputer(
                    missing_values=np.nan, strategy="mean"
                )  # XXX settings

                real_values = self[attr].values.astype(float)

                if not np.isnan(real_values).all():
                    transformed_data = imputer.fit_transform(
                        real_values.reshape(-1, 1)
                    ).reshape(-1, 1)
                else:
                    transformed_data = np.zeros(real_values.reshape(-1, 1).shape)
                columns_index = pd.MultiIndex.from_product(
                    [[attr], [values]], names=["0", "1"]
                )
            elif values.lower() == "string":
                imputer = SimpleImputer(missing_values=None, strategy="most_frequent")
                imp_data = imputer.fit_transform(
                    self[attr].values.reshape(-1, 1)
                ).reshape(-1, 1)
                transformed_data = enc.transform(imp_data).toarray()
                columns_index = pd.MultiIndex.from_product(
                    [[attr], range(transformed_data.shape[1])], names=["0", "1"]
                )
            else:
                raise ValueError(attr, values)

            new = pd.DataFrame(transformed_data, columns=columns_index)
            if encoded_dataframe.empty:
                encoded_dataframe = new
            else:
                encoded_dataframe = encoded_dataframe.join(new)

        # here we convert result into DataFrameArff to keep arff metadata TODO: Here starts the newer implememtation
        # new_columns = convert_multiindex_to_index(encoded_dataframe.columns)
        #
        # arff_data = ArffData(relation=self._arff_data["relation"] + "_binarized",
        #                      description=self._arff_data["description"],
        #                      attributes=[(name, 'NUMERIC') for name in new_columns],
        #                      data=encoded_dataframe.values)
        # return DataFrameArff(arff_data=arff_data.__dict__)

        return encoded_dataframe # TODO: This was old return value

    def binarize_cat_feats_and_normalize(self, keep_class: bool=False)->'DataFrameArff':
        bin_df: pd.DataFrame = self._binarize_categorical_values()
        #print("ARFF DATA", self._arff_data.keys())

        new_columns = convert_multiindex_to_index(bin_df.columns)
        _relation = self._arff_data["relation"] + "-binarized-normalized"
        _attributes = [(name, 'NUMERIC') for name in new_columns]
        # if we want to keep the class column among the binarized data, we have to add it back,
        # because it is not retained during binarisation
        if keep_class:
            _relation += "with-class"
            _attributes += self._arff_data["attributes"][-1]
            bin_df[self.columns[-1]] = self[self.columns[-1]]
        else:
            _relation += "-class-removed"
        arff_data = ArffData(relation=_relation,
                             description="", #self._arff_data["description"],  # TODO: description is truncated here
                             attributes=_attributes,
                             data=bin_df.values)
        return DataFrameArff(arff_data=arff_data)


    def add_index_column(self):
        if ID_NAME not in self.columns:
            self.insert(loc=0, column=ID_NAME, value=self.index.values)
            self._arff_data["attributes"].insert(0, (ID_NAME, "NUMERIC"))
        return self

    def get_dataframe_without_id(self):
        dataframe_without_id = DataFrameArff(
            self.loc[:, self.columns != ID_NAME].values,
            columns=self.columns[self.columns != ID_NAME],
        )
        dataframe_without_id._arff_data = {
            **self._arff_data,
            "attributes": [x for x in self._arff_data["attributes"] if x[0] != ID_NAME],
        }
        return dataframe_without_id

    def apply_outlier_detector(self, detector_schema: OutlierDetectorSchema):
        detector = DETECTORS[detector_schema.name](**detector_schema.parameters)



        time_start = resource.getrusage(resource.RUSAGE_SELF)[0]
        time_start_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

        detector.compute_scores(self.get_dataframe_without_id(), self[self.columns[-1]])

        time_end = resource.getrusage(resource.RUSAGE_SELF)[0]
        time_end_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

        od_time = (time_end - time_start) + (time_end_children - time_start_children)

        new_frame = DataFrameArff(self.values, columns=self.columns)
        new_frame._arff_data = self._arff_data

        new_frame.insert(
            loc=len(self.columns) - 1, column=OD_VALUE_NAME, value=detector.values
        )
        new_frame._arff_data["attributes"].insert(
            -1, (OD_VALUE_NAME, detector.data_type)
        )

        return new_frame, od_time

    def apply_custom_feature_selector(self, f_selector_chema: CustomFSSchema):
        f_selector: AbstractFeatureSelector = F_SELECTORS[f_selector_chema.name](**f_selector_chema.parameters)

        time_start = resource.getrusage(resource.RUSAGE_SELF)[0]
        time_start_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

        new_frame = f_selector.select_features(self.get_dataframe_without_id(), self[self.columns[-1]])


        time_end = resource.getrusage(resource.RUSAGE_SELF)[0]
        time_end_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

        fs_time = (time_end - time_start) + (time_end_children - time_start_children)

        new_frame_arff = DataFrameArff(new_frame.values, columns=new_frame.columns)
        new_frame_arff._arff_data = self._arff_data
        attribute_set = set(new_frame.columns)
        # copy only those arff attributes that were selected, presuming that column names are same as arff attribute names
        new_frame._arff_data["attributes"] = [
            x
            for x in new_frame._arff_data["attributes"]
            if x[0] in attribute_set
        ]

        return new_frame_arff, fs_time

    def select_features_with_sklearn(self, selector: _BaseFilter, leave_binarized: bool):
        #colnames = self.columns

        # make sure that ID column does not compromise the feature selection
        if ID_NAME in self.columns:
            self.drop(ID_NAME, 1, inplace=True)
            self._arff_data["attributes"] = [x for x in self._arff_data["attributes"] if x[0] != ID_NAME]
            #colnames = self.columns
            print(ID_NAME, "deleted for dataset: ", self._arff_data["relation"])
            print(self)
        bin_df: pd.DataFrame = self._binarize_categorical_values()

        # split data and classes. We rely on the fact that classes are in the last column
        x = bin_df # Binarisation already gets rid of class #.loc[:, self.columns[:-1]]
        y = self.loc[:, self.columns[-1]]
        # another score functions are: f_classif, mutual_info_classif

        time_start = resource.getrusage(resource.RUSAGE_SELF)[0]
        time_start_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]

        # fit selector to the dataset (this basically looks at the dataset and identifies useful features)
        print("TRAIN DATA", x, sep="\n")
        print("TEST DATA", y, sep="\n")
        selector.fit(x, y)

        selected_features_indexes = selector.get_support()
        # print(selected_features_indexes)

        # here we are indexing the binarized, filtered dataset by a list of bools.
        trans_df: pd.DataFrame = x.iloc[:, selected_features_indexes]

        #print(trans_df)
        nmi: pd.MultiIndex = trans_df.columns

        if not leave_binarized:
            selected_feature_indexes_set = set()
            selected_feature_indexes_list = []
            for code in nmi.codes[0]:
                if code not in selected_feature_indexes_set:
                    selected_feature_indexes_list.append(code)
                selected_feature_indexes_set.add(code)
            # here we actually add the "classes" column back into the list of selected features
            selected_feature_indexes_list.append(len(self.columns) - 1)
            # also, we need to sort the indexes list, bcs the order is significant when determining which column has
            # which values
            selected_feature_indexes_list.sort()
            # print(selected_feature_indexes_list)
            final_df = self.iloc[:, selected_feature_indexes_list]

            # change the ARFF data so that they contain updated data
            selected_columns_set = set(final_df.columns)
            # obtain original arff data
            arff_data = self.arff_data()
            arff_data.attributes = [x for x in arff_data.attributes if x[0] in selected_columns_set]
            # adding the "arff_data" keyword bypasses the super.__init__() method in DataFrameArff, so we need to overwrite
            # the vlaues inside the arff_data themselves.
            arff_data.data = final_df.values
            new_frame_arff: DataFrameArff = DataFrameArff(arff_data=arff_data)

        else:
            # new_columns = convert_multiindex_to_index(nmi)
            #
            # arff_data = {
            #     "relation": self._arff_data["relation"],
            #     "description": "", # self._arff_data["description"], # description takes space --> unnecessary
            #     "attributes": [(name, 'NUMERIC') for name in new_columns],
            #     "data": trans_df.values
            # }
            trans_df.insert(value=self[self.columns[-1]], column=self.columns[-1], loc=len(trans_df.columns))
            new_frame_arff: DataFrameArff = add_arff_metadata_to_pandas_dataframe(trans_df, ArffData(**self._arff_data))

        # conclude time (resource) consumption
        time_end = resource.getrusage(resource.RUSAGE_SELF)[0]
        time_end_children = resource.getrusage(resource.RUSAGE_CHILDREN)[0]
        fs_time = (time_end - time_start) + (time_end_children - time_start_children)

        return new_frame_arff, fs_time

    def select_by_index(self, index: np.array):
        dataframe = self.iloc[index]
        arff_dataframe = DataFrameArff(dataframe.values, columns=self.columns)
        arff_dataframe._arff_data = self._arff_data

        return arff_dataframe

    def select_by_od_quantile(self, quantile, reverse=False):
        value = self[OD_VALUE_NAME]

        dataframe = self.iloc[np.sort(np.argsort(value, kind='quicksort')[:round(quantile * len(value))]), :] \
            if not reverse \
            else self.iloc[np.sort(np.argsort(-value, kind='quicksort')[:round(quantile * len(value))]), :]

        arff_dataframe = DataFrameArff(dataframe.values, columns=self.columns)
        arff_dataframe._arff_data = self._arff_data

        return arff_dataframe


class DataLoader:
    def __init__(self, data_path: str, regex: str = r".*"):
        self._reg = re.compile(regex)
        self.file_paths: List[str] = []
        if os.path.isdir(data_path):
            files = (
                x
                for x in os.listdir(data_path)
                if x.endswith(".arff") and self._reg.match(x)
            )
            for file_name in files:
                self.file_paths.append(os.path.join(data_path, file_name))
        elif (
            os.path.isfile(data_path)
            and data_path.endswith(".arff")
            and self._reg.match(data_path)
        ):
            self.file_paths.append(data_path)

        self.file_paths = sorted(self.file_paths, key=lambda x: os.path.getsize(x))

    @staticmethod
    def _load_data_file(file_path: str):
        raise NotImplementedError()

    @staticmethod
    def _load_arff_file(file_path: str) -> DataFrameArff:
        print("trying to open ", end="")
        with open(file_path) as arff_file:
            print("and load and arff", os.path.basename(file_path), end=" ")
            data = arff.load(arff_file)
            print("and then close it ", end="")
            arff_file.close()
            print("and return", end="")
            return DataFrameArff(arff_data=data)


    def load_files(self):
        if not self.file_paths:
            raise RuntimeError(
                "No .arff detected. Please specify a correct path and unzip data file."
            )
        for file_path in self.file_paths:
            yield self._load_arff_file(file_path)


def convert_multiindex_to_index(mi: pd.MultiIndex) -> [str]:
    # setup dictionary that will contain column names of column created by binarisation in lists under keys by their
    # original columns names before binarisation
    columns = {}
    for i in range(len(mi)):
        original_colname = mi.levels[0][mi.codes[0][i]]  # this extracts the original name of column
        catname = mi.levels[1][mi.codes[1][i]]  # this extracts the subcolumn names (categories)
        if original_colname not in columns:
            # create an entry for a column name
            columns[original_colname] = []
        # append to the list of subcolumn names (categories) of the column
        columns[original_colname].append(catname)

    # init a list of new columns
    new_columns = []
    for original_colname in columns.keys():
        # the binarisation leaves names of WEKA data types instead of nominal values for columns representing
        # non-categorical values. To avoid  unnecessary renaming, we actually check for those specific names.
        if len(columns[original_colname]) == 1 and columns[original_colname][0] in WEKA_DATA_TYPES:
            new_columns.append(original_colname)
        else:
            for subcolname in columns[original_colname]:
                new_columns.append(original_colname + "_" + subcolname)
    return new_columns


def add_arff_metadata_to_pandas_dataframe(df: pd.DataFrame, arff_data: ArffData)->DataFrameArff:
    if isinstance(df.columns, pd.MultiIndex):
        new_columns = convert_multiindex_to_index(df.columns)
    else:
        new_columns = df.columns

    arff_data = ArffData(relation=arff_data.relation,
                         description="",#arff_data.description,  # description is truncated for space preservation
                         attributes=[(name, 'NUMERIC') for name in new_columns],
                         data=df.values)
    return DataFrameArff(arff_data=arff_data.__dict__)
