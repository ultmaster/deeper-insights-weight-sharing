import os
from argparse import ArgumentParser

import numpy as np


def tuple_four_display(tp):
    ret = "${:.4f} _{{\\pm {:.4f}}}$".format(tp[0], tp[1])
    return ret


def tuple_two_display(tp):
    return "${:.4f}$ & ${:.4f}$ & ${:.4f}$".format(tp[0], tp[2], tp[3])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--metric", action="append")
    parser.add_argument("--folder", action="append")
    parser.add_argument("--show-max-min", action="store_true", default=False)
    args = parser.parse_args()
    ret = []
    print("\\toprule")
    if args.show_max_min:
        print("Mean & Max & Min \\\\")
    else:
        print("".join([" & " + metric for metric in args.metric]) + " \\\\")
    print("\\midrule")
    for folder in args.folder:
        ret = folder
        for metric in args.metric:
            path = os.path.join(folder, "METRICS-" + metric + ".txt")
            nt = np.loadtxt(path)
            ret += " & " + (tuple_two_display(nt) if args.show_max_min else tuple_four_display(nt))
        ret += " \\\\"
        print(ret)
    print("\\bottomrule")
