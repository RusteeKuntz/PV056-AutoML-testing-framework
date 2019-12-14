import numpy as np

from sklearn.ensemble import RandomForestClassifier
import pyximport; pyximport.install()

class RFOEXMetric:

    def countRFOEX(self, df, classes, super):

        inst = len(df)

        estimator = RandomForestClassifier(**super.settings)
        estimator.fit(df.values, classes)

        cdef int i, j

        trees = estimator.estimators_
        cdef short[:,:] matrix = np.empty((inst, inst), np.short)

        cdef short[:,:] predicted = np.empty((len(trees), inst), np.short)

        for i in range(len(trees)):
            pred = trees[i].predict(df)
            for j in range(inst):
                predicted[i][j] = pred[j]

        for i in range(inst):
            for j in range(i, inst):
                sim = 0
                for t in range(len(trees)):
                    if predicted[t][i] == predicted[t][j]:
                        sim += 1
                matrix[i, j] = sim
                matrix[j, i] = sim

        _, cls_num = np.unique(classes, return_inverse=True)
        cdef int[:] clss = cls_num.astype(int)

        #cdef int cl
        cls_indices = {}
        for cls in np.unique(clss):
            cls_indices[cls] = [i for i in range(len(df)) if clss[i] == cls]

        cdef double[:] proxsuminv = np.empty(inst)
        cdef double[:] wrong_cl = np.empty(inst)
        cdef double[:] general_out = np.empty(inst)

        cdef double[:] row
        cdef int s

        for i in range(inst):

            row = matrix.base[i] / len(trees)

            # ProxSumInverse
            indices = cls_indices[clss[i]]
            proxsuminv[i] = inst / sum([row[j] for j in indices])

            # TopC instances
            num_inst = len(cls_indices[clss[i]])
            top_c = np.argsort(np.multiply(row, -1))[:num_inst]

            # P_wrong_cl
            s = 0
            for j in top_c:
                if cls_num[i] != cls_num[j]:
                    s += 1
            wrong_cl[i] = s / num_inst

            # P_general_out
            general_out[i] = (num_inst - sum(row[j] for j in top_c)) / num_inst

        means = {}
        meandevs = {}
        for cls in np.unique(clss):
            means[cls] = np.mean(proxsuminv.base[cls_indices[cls]])
            meandevs[cls] = np.mean(abs(proxsuminv[cls_indices[cls]] - means[cls]))
        const = np.max(proxsuminv) / 4

        fo1 = [(proxsuminv[i] - means[clss[i]]) / meandevs[clss[i]] if meandevs[clss[i]] != 0 else 0 for i in range(inst)]
        fo2 = wrong_cl * const
        fo3 = general_out * const

        values = np.add(np.add(fo1, fo2), fo3)

        return values