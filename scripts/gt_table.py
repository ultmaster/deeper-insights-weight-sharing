import numpy as np
from scipy.stats import rankdata


gt_mean = np.loadtxt("assets/gt_mean.txt")
gt_std = np.loadtxt("assets/gt_std.txt")
gt_rank = rankdata(-gt_mean)
lines = 32
result = [[] for _ in range(lines)]
for i in range(64):
    result[i % lines].extend([
        "%d%d%d" % (i // 16 + 1, (i // 4) % 4 + 1, i % 4 + 1),
        "%.2f" % (gt_mean[i] * 100),
        "%.2f" % (gt_std[i] * 100),
        "%d" % (gt_rank[i])
    ])
for r in result:
    print(" & ".join(r) + " \\\\")