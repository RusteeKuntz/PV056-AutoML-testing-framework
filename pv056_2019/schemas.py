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
class ScikitCommandSchema(FSStepSchema):
    source_library = SCIKIT
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
    blacklist_file_path: str
    n_jobs: int = 5
    timeout: int = 1800
    selection_methods: List[FSStepSchema]

# importing stuff from feature_selection to avoid circular dependency error
from pv056_2019.feature_selection import F_SELECTORS
from pv056_2019.outlier_detection import DETECTORS