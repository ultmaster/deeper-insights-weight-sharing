import functools
import os
from argparse import ArgumentParser

import networkx
import numpy as np

from visualize import heatmap


class MatchingClustering(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        total = len(X)
        grouping = [{i} for i in range(total)]
        while len(grouping) > self.n_clusters:
            gx = networkx.Graph()
            gx.add_nodes_from(list(range(len(grouping))))
            for i in range(len(grouping)):
                for j in range(i + 1, len(grouping)):
                    w = sum([X[x][y] + 1 for x in grouping[i] for y in grouping[j]])
                    gx.add_edge(i, j, weight=w + 1)
            ret = networkx.algorithms.max_weight_matching(gx, maxcardinality=True)
            ret = sorted(list(ret))
            new_grouping = [grouping[a] | grouping[b] for a, b in ret]
            grouping = new_grouping
        if len(grouping) != self.n_clusters:
            raise ValueError("Cannot satisfy need: splitting to {} clusters.".format(self.n_clusters))
        ret = np.zeros(total, dtype=np.int)
        for i, g in enumerate(grouping):
            ret[list(g)] = i
        return ret


def main():
    parser = ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument("--num_classes", default=2, type=int)

    args = parser.parse_args()

    cor_matrix = np.loadtxt(os.path.join(args.dir, "corr_heatmap.txt"))
    grouping = np.loadtxt(os.path.join(args.dir, "group_info.txt"), dtype=np.int)
    group_number = np.max(grouping) + 1
    result_grouping = np.zeros(len(grouping), dtype=np.int)

    base = 0
    for i in sorted(list(range(group_number)), key=lambda d: np.sum(grouping == d)):
        cur_index = np.where(grouping == i)[0]
        cor_matrix_grp = cor_matrix[cur_index][:, cur_index]
        # print(cor_matrix_grp)
        model = MatchingClustering(n_clusters=args.num_classes)
        if len(cur_index) < args.num_classes:
            result_grouping[cur_index] = np.arange(len(cur_index)) + base
            print("Group {}: Too small")
            base += len(cur_index)
        else:
            predict = model.fit_predict(1 - cor_matrix_grp)
            print("Group {}: {}".format(i, predict.tolist()))
            for j in range(args.num_classes):
                result_grouping[cur_index[predict == j]] = base + j
            base += args.num_classes
        heatmap(cor_matrix_grp, filepath=os.path.join("debug", "heatmap_{}".format(i)))

    print(result_grouping.tolist())


if __name__ == "__main__":
    main()
