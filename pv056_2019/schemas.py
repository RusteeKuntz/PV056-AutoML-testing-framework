import re

from typing import List, Union

from pydantic import BaseModel, validator

from pv056_2019.utils import NONE_STR, WEKA, SCIKIT, CUSTOM


class SplitterSchema(BaseModel):
    train_split_dir: str
    test_split_dir: str
    data_path: str
    k_of_folds: int = 5
    m_of_repeats: int = 1
    random_state: int = 42

    @validator("m_of_repeats")
    def n_jobs_validator(cls, value):
        if value < 1:
            raise ValueError("m_of_repeats must be greater than 0")
        return value

    @validator("k_of_folds")
    def k_of_folds_validator(cls, value):
        if value <= 1:
            raise ValueError("k_of_folds must be greater than 1")
        return value


class OutlierDetectorSchema(BaseModel):
    name: str
    parameters: dict

    @validator("name")
    def detector_name(cls, value):
        if value not in DETECTORS.keys() and value != NONE_STR:
            raise ValueError(
                "Detector {} is not supported. Supported detectors are: {}".format(
                    value, ", ".join(DETECTORS.keys())
                )
            )
        return value


class ODStepConfigSchema(BaseModel):
    #train_split_dir: str
    od_methods: List[OutlierDetectorSchema]
    train_od_dir: str
    n_jobs: int = 1
    times_output: str

    @validator("n_jobs")
    def n_jobs_validator(cls, value):
        if value < 1:
            raise ValueError("n_jobs must be greater than 0")

        return value


class RemoveOutliersConfigSchema(BaseModel):
    #test_split_dir: str
    #train_od_dir: str
    percentage: Union[float, List[float]]
    train_removed_dir: str
    #reverse: bool = False
    #keep_original: bool = True

    @validator("percentage")
    def percentage_validator(cls, value):
        if abs(value) >= 100:
            raise ValueError("Percentage of removed outliers must be between -100 and 100.")
        if value == 0:
            raise ValueError("Don't use zero for creating baseline. If you want to remove no outliers, use the original files.")

        return value


class FilterSchema(BaseModel):
    name: str
    args: List[str] = []


class ClassifierSchema(BaseModel):
    class_name: str
    args: List[str] = []
    filters: List[FilterSchema] = []

    @property
    def name(self):
        return self.class_name.split(".")[-1]


class RunClassifiersCongfigSchema(BaseModel):
    output_folder: str
    weka_jar_path: str
    classifiers: List[ClassifierSchema]
    n_jobs: int = 1
    times_output: str = "clf_times"
    blacklist_file: str
    timeout: int

    @validator("n_jobs")
    def n_jobs_validator(cls, value):
        if value < 1:
            raise ValueError("n_jobs must be greater than 0")

        return value


class StatisticsSchema(BaseModel):
    results_dir: str
    #od_times_path: str
    #clf_times_path: str
    output_table: str
    aggregate: bool = True
    pattern: str = ".*"


class FSStepSchema(BaseModel):
    source_library: str


class CommandSchema(BaseModel):
    """ This represents the schema of configuration for an arbitrary feature selection class invokation """
    name: str
    parameters: dict = {}


# TODO: Create generic scikit method calling mechanism
class ScikitFSSchema(FSStepSchema):
    source_library = SCIKIT
    leave_attributes_binarized: bool
    fs_method: CommandSchema
    score_func: CommandSchema = None


class CustomFSSchema(FSStepSchema, CommandSchema):
    source_library = CUSTOM
    name: str
    parameters: dict = {}

    @validator("name")
    def f_selector_name(cls, value):
        if value not in F_SELECTORS.keys():
            raise ValueError(
                "Feature selector {} is not supported. Supported selectors are: {}".format(
                    value, ", ".join(DETECTORS.keys())
                )
            )

        return value

class WekaFSFilterConfigurationSchema(FSStepSchema):
    source_library = WEKA
    eval_class: CommandSchema
    search_class: CommandSchema
    #n_folds: int = 0
    #cv_seed: int = 123

    #@validator("n_folds")
    #def folds_validator(cls, value):
    #    if value <= 1:
    #        raise ValueError("number of folds must be greater than 1")
    #    return value



class FeatureSelectionStepSchema(BaseModel):
    weka_jar_path: str
    output_folder_path: str
    blacklist_file_path: str = None
    n_jobs: int = 5
    timeout: int = 1800
    selection_methods: List[dict]


class GraphStepSchema(BaseModel):
    sort_by_column: Union[str, List[str]] = None
    separate_graphs_for_different_values_in_column: Union[str, List[str]] = None
    dpi: int = 400  # dots per inch, resolution


class GraphScatterStepSchema(GraphStepSchema):
    col_examined: str = "accuracy"
    col_related: Union[str, List[str]]
    col_grouped_by: Union[str, List[str]]
    legend_title: str
    title: str
    x_title: str
    y_title: str
    max_y_val: Union[float, int] = None
    min_y_val:Union[float, int] = None
    convert_col_related_from_json: bool = True
    extract_col_grouped_by: Union[str, List[str]] = None
    extract_col_related: Union[str, List[str]] = None



class GraphBoxStepSchema(GraphStepSchema):
    col_examined: str = "accuracy"
    col_related: Union[str, List[str]]
    title: str
    x_title: str
    y_title: str
    sort_func_name: str = None  # label, mean, inv_mean, median
    min_y_val: Union[float, int] = None
    max_y_val: Union[float, int] = None
    convert_col_related_from_json: bool = True
    extract_col_related: str = None
    show_fliers: bool = True
    dpi: int = 600  # dots per inch, resolution



# importing stuff from feature_selection to avoid circular dependency error
from pv056_2019.feature_selection import F_SELECTORS
from pv056_2019.outlier_detection import DETECTORS