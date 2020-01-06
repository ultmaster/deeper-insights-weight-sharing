import numpy as np


def get_all_architecture_indices_combination(n_prims, n_edges):
    def dfs(now):
        nonlocal ret
        if len(now) == n_edges:
            ret.append(tuple(now))
        else:
            for i in range(n_prims):
                dfs(now + [i])

    ret = []
    dfs([])
    return ret


def find_sim(arr):
    ret = []
    for i, a in enumerate(arr):
        for j, b in enumerate(arr):
            ret.append(np.sum(np.array(a) == np.array(b)) / 3)
    return sum(ret) / len(ret)


all_archits = get_all_architecture_indices_combination(4, 3)
for m in [1, 2, 4, 8, 16, 32, 64]:
    each = len(all_archits) // m
    print(m, "%.4f" % find_sim(all_archits[:each]))
