import functools
import logging
import math
import os
import pickle

import seaborn as sns
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

__logger__ = logging.getLogger("visualizer")
__logger__.addHandler(logging.StreamHandler())
__logger__.setLevel(logging.WARNING)


def generate_full_path(filepath):
    if "analysis" not in filepath:
        return filepath + ".pdf"
    file_fullname = os.path.relpath(filepath, "analysis")
    file_fullname = file_fullname.replace("/", "_").replace("\\", "_").replace(".", "_")
    return os.path.join(os.path.dirname(filepath), file_fullname + ".pdf")


def generate_checkpoint_path(filepath):
    file_dir, file_name = os.path.dirname(filepath), "REPLAY_" + os.path.basename(filepath) + ".pkl"
    return os.path.join(file_dir, file_name)


class MultiPageContext(PdfPages):
    def __init__(self, filename, **kwargs):
        super().__init__(generate_full_path(filename), **kwargs)
        self.checkpoint_path = generate_checkpoint_path(filename)
        self.checkpoints = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        with open(self.checkpoint_path, "wb") as fp:
            pickle.dump(self.checkpoints, fp)
        super().__exit__(exc_type, exc_val, exc_tb)


def get_font_dict(fontsize, prefix="font"):
    if fontsize is None:
        return dict()

    return {prefix + "family": "Times New Roman",
            prefix + "size": fontsize,
            prefix + "weight": 300}


def init(figsize=(20, 10), xlabel="", ylabel="", title="", fontsize=None, **kwargs):
    plt.figure(figsize=figsize)
    plt.grid(linestyle="--")
    fontdict = get_font_dict(fontsize)
    plt.xticks(**fontdict)
    plt.yticks(**fontdict)

    if title:
        plt.title(title, **fontdict)

    if xlabel:
        plt.xlabel(xlabel, **fontdict)
    if ylabel:
        plt.ylabel(ylabel, **fontdict)


def finalize(filepath=None, inverse_y=False, context=None, labels=None, fontsize=None, legend_loc="best",
             legend_labelspacing=0.5, legend_borderpad=0.4, legend_borderaxespad=0.5, margins=None, **kwargs):
    ax = plt.gca()
    if margins is not None:
        ax.margins(*margins)
    if inverse_y:
        ax.set_ylim(ax.get_ylim()[::-1])
    if labels is not None:
        ax.legend(prop=get_font_dict(fontsize, prefix=""), loc=legend_loc, labelspacing=legend_labelspacing,
                  borderpad=legend_borderpad, borderaxespad=legend_borderaxespad)
    if context is not None:
        context.savefig(bbox_inches="tight")
    else:
        assert filepath is not None
        plt.savefig(generate_full_path(filepath), format="pdf", bbox_inches="tight")
        __logger__.info("Visualization writing to {}".format(filepath))
    plt.close()


def save_checkpoint(func, *args, **kwargs):
    ckpt = {
        "func": func.__name__,
        "args": args,
        "kwargs": {k: v for k, v in kwargs.items() if k != "context"}
    }
    if "filepath" not in kwargs:
        kwargs["context"].checkpoints.append(ckpt)
    else:
        with open(generate_checkpoint_path(kwargs["filepath"]), "wb") as fp:
            pickle.dump(ckpt, fp)


def init_and_finalize(func):
    @functools.wraps(func)
    def foo(*args, **kwargs):
        save_checkpoint(func, *args, **kwargs)
        init(**kwargs)
        func(*args, **kwargs)
        finalize(**kwargs)
    return foo


@init_and_finalize
def boxplot(data, xticklabels=None, **kwargs):
    plt.boxplot(data)
    if xticklabels is not None:
        ax = plt.gca()
        ax.set_xticklabels(xticklabels)


@init_and_finalize
def scatterplot(data, **kwargs):
    for d in data:
        x, y = d[:, 0], d[:, 1]
        plt.scatter(x, y)


@init_and_finalize
def lineplot(data, color=None, alpha=1.0, markers=None, labels=None, fmt="-", cutoff=None, x_offset=None,
             markersize=None, data_cutoff=None, **kwargs):
    # data should be three dimensional
    # number of lines * number of points * 2
    if data_cutoff is not None:
        data = data[:data_cutoff]
    if color is not None:
        assert len(data) == len(color)
        cmap = cm.get_cmap("viridis")
    if isinstance(fmt, str):
        if markers is not None:
            fmt += "D"
        fmt = [fmt] * len(data)
    for i, d in enumerate(data):
        kwargs = {"alpha": alpha}
        if color is not None:
            if isinstance(color[i], str):
                kwargs.update({"color": color[i]})
            else:
                kwargs.update({"color": cmap(color[i])})
        if markers is not None:
            kwargs.update({"markevery": markers[i]})
        if labels is not None and i < len(labels):
            kwargs.update({"label": labels[i]})
        if markersize is not None:
            kwargs.update({"markersize": markersize})
        if cutoff is not None:
            d = d[-cutoff:]
        if x_offset is not None:
            d[:, 0] += x_offset
        plt.plot(d[:, 0], d[:, 1], fmt[i], **kwargs)


@init_and_finalize
def errorbar(data, **kwargs):
    for i, d in enumerate(data):
        plt.errorbar(d[:, 0], d[:, 1], d[:, 2], fmt="-o")


@init_and_finalize
def heatmap(data, **kwargs):
    ax = sns.heatmap(data, linewidths=1)
    ax.set_aspect("equal")
    a, b = ax.get_ylim()
    ax.set_ylim(math.ceil(a), math.floor(b))


def replay(checkpoint, **kwargs):
    if isinstance(checkpoint, str):
        checkpoint = [checkpoint]
    func_name, combined_args, combined_kwargs = None, None, None
    for filename in checkpoint:
        with open(filename, "rb") as fp:
            ckpt = pickle.load(fp)
        if isinstance(ckpt, list):
            ckpt = ckpt[kwargs["page"]]
        if func_name is None:
            func_name = ckpt["func"]
            combined_args = ckpt["args"]
            combined_kwargs = ckpt["kwargs"]
        else:
            assert func_name == ckpt["func"]
            for a, b in zip(combined_args, ckpt["args"]):
                if isinstance(a, list):
                    a.extend(b)
            for k, v in combined_kwargs.items():
                if isinstance(v, list):
                    v.extend(ckpt["kwargs"][k])
    combined_kwargs.update(kwargs)
    eval(func_name)(*combined_args, **combined_kwargs)
