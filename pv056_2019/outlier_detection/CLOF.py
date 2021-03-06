import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class CLOFMetric:

    @staticmethod
    def compute_values(df, classes, **kwargs):

        if "alfa" in kwargs:
            alfa = float(kwargs["alfa"])
            del kwargs["alfa"]
        else:
            alfa = 0.75

        if "beta" in kwargs:
            beta = float(kwargs["beta"])
            del kwargs["beta"]
        else:
            beta = 0.25

        _, cls_num = np.unique(classes, return_inverse=True)
        clss = cls_num.astype(int)

        cls_indices = {}
        noncls_indices = {}
        for cls in np.unique(clss):
            cls_indices[cls] = [i for i in range(len(df)) if clss[i] == cls]
            noncls_indices[cls] = [i for i in range(len(df)) if clss[i] != cls]

        lof = LocalOutlierFactor(**kwargs)
        lof.fit(df.values)
        lofn = LocalOutlierFactor(**kwargs, novelty=True)

        same_lof = np.empty(len(df))
        other_lof = np.empty(len(df))
        all_lof = lof.negative_outlier_factor_
        for cls in np.unique(clss):
            ind = cls_indices[cls]
            nind = noncls_indices[cls]
            lof.fit(df.iloc[ind])
            same_lof[ind] = lof.negative_outlier_factor_
            lofn.fit(df.iloc[nind])
            for i in ind:
                v = lofn.decision_function([df.iloc[i]])
                other_lof[i] = 1 / v if v != 0 else 10

        values = -1 * (same_lof + alfa * other_lof + beta * all_lof)

        return values
