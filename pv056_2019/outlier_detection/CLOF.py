import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


class CLOFMetric:

    def __init__(self, **kwargs):
        self.alfa = float(kwargs["alfa"]) if ("alfa" in kwargs) else 0.75
        self.beta = float(kwargs["beta"]) if ("beta" in kwargs) else 0.25

    def compute_values(self, df, classes):
        unique_classes = np.unique(classes)

        _, cls_num = np.unique(classes, return_inverse=True)
        clss = cls_num.astype(int)

        cls_indices = {}
        noncls_indices = {}
        for cls in np.unique(clss):
            cls_indices[cls] = [i for i in range(len(dataframe)) if clss[i] == cls]
            noncls_indices[cls] = [i for i in range(len(dataframe)) if clss[i] != cls]


        lof = LocalOutlierFactor()
        lof.fit(df.values)


        same_lof = np.empty(len(df))
        other_lof = np.empty(len(df))
        all_lof = lof._decision_function(df.values)
        for cls in np.unique(clss):
            ind = cls_indices[cls]
            nind = noncls_indices[cls]
            lof.fit(df.iloc[ind])
            same_lof[ind] = lof._decision_function(df.iloc[ind].values)
            for i in ind:
                lof.fit(df.iloc[nind].append(df.iloc[i]).values)
                other_lof[i] = lof._decision_function(df.iloc[ind].values)








        clf = LocalOutlierFactor()
        clf.fit(dataframe.values)
        df_lof = clf._decision_function(dataframe.values)

        class_separated_lof = []
        for df in class_separated_dataframes:
            class_separated_lof.append(clf._decision_function(df.values))

        values = [0] * len(dataframe)

        for index, row in dataframe.iterrows():
            class_column = dataframe.columns.values[-1]
            row_class_index = np.where(unique_classes == row[class_column])[0][0]

            other_classes_dataframes_indexes = list(range(len(class_separated_dataframes)))
            other_classes_dataframes_indexes.remove(row_class_index)
            other_classes_dataframe = pd.DataFrame()
            other_classes_dataframe = other_classes_dataframe.append(
                [class_separated_dataframes[i] for i in other_classes_dataframes_indexes]
            )
            other_classes_dataframe = other_classes_dataframe.append(dataframe.loc[index])
            other_classes_lof = clf._decision_function(other_classes_dataframe.values)

            row_location = class_separated_dataframes[row_class_index].index.get_loc(index)
            same_lof = class_separated_lof[row_class_index][row_location]
            other_lof = other_classes_lof[-1]
            all_lof = df_lof[index]

            values[index] = same_lof + self.alfa * (1 / other_lof) + self.beta * all_lof

        return values
