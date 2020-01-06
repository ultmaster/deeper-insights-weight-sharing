""" CNN for architecture search """
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import genotypes as gt
from models.search_cells import SearchCell


class SearchCNN(nn.Module):
    """
    Search CNN model

    A simplified model of the original darts
    """

    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3, bn_momentum=0.1):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur, momentum=bn_momentum),
            nn.Conv2d(C_cur, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur, momentum=bn_momentum)   # modification: twice conv2d
        )

        # we discard the skip connection between cells
        # [!] C_p is output channel size, but C_cur is input channel size.
        C_p, C_cur = C_cur, C

        self.cells = nn.ModuleList()
        for i in range(n_layers):
            # No reduction
            cell = SearchCell(n_nodes, C_p, C_cur, bn_momentum=bn_momentum)
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_p = C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal):
        s = self.stem(x)

        for cell in self.cells:
            s = cell(s, weights_normal)

        out = self.gap(s)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3, bn_momentum=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        self.device_ids = list(range(torch.cuda.device_count()))

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(torch.zeros(i + 1, n_ops)))
        self.generate_and_fill_alphas("random")

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes, stem_multiplier, bn_momentum)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        return self.net(x, weights_normal)
        # there seems still something wrong with multiple GPU support

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        concat = range(1, 1 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def generate_and_fill_alphas(self, rule, **kwargs):
        """
        :param rule:
        If rule is random, generate as originally did
        If rule is fixed, generate the alphas according to `select`, with length (n_ops) * (n_ops + 1) / 2
        :param kwargs:
        """
        if rule == "random":
            for i in range(self.n_nodes):
                self.alpha_normal[i].data = torch.randn_like(self.alpha_normal[i].data) * 1E-3
        elif rule == "fixed":
            assert len(kwargs["select"]) == (self.n_nodes + 1) * self.n_nodes // 2
            select = kwargs["select"]
            k = 0
            for i in range(self.n_nodes):
                self.alpha_normal[i].data.fill_(float('-inf'))
                for j in range(i + 1):
                    self.alpha_normal[i].data[j][select[k]] = 0.
                    k += 1
        else:
            raise NotImplementedError

    def generate_unshared_mask(self, shared_rule, **kwargs):
        """
        :param shared_rule: Rule is, semantically, to share which part
        If rule is none, share none
        If rule is all, share all
        If rule is prefix, share prefix, accepting kwargs: k (how many layers)
        If rule is not_dag, share all but the edges introduced by dags
        :param kwargs:
        :return: a list of keys to trace in state_dict()
        """
        if shared_rule == "all":
            return []
        # alpha is always ignored: always shared
        # alpha is always reloaded by trainer when training
        all_keys = [t for t in self.state_dict().keys() if not t.startswith("alpha")]
        if shared_rule in ["none", "group"]:
            return all_keys
        if shared_rule == "not_dag":
            return list(filter(lambda t: "dag" in t, all_keys))
        if shared_rule == "prefix":
            k = kwargs["k"]
            shared_layer_prefix = ["net.stem"]
            for i in range(k):
                shared_layer_prefix.append("net.cells.%d" % i)
            unshared_layers = []
            for key in all_keys:
                if not any(key.startswith(prefix) for prefix in shared_layer_prefix):
                    unshared_layers.append(key)
            return unshared_layers
        if shared_rule == "keyword":
            unshared_layers = []
            keywords = kwargs["kw"]
            for key in all_keys:
                if not any(kw in key for kw in keywords):
                    unshared_layers.append(key)
            return unshared_layers
        if shared_rule == "keyword_unshare":
            unshared_layers = []
            keywords = kwargs["kw"]
            for key in all_keys:
                if any(kw in key for kw in keywords):
                    unshared_layers.append(key)
            return unshared_layers
        raise NotImplementedError

    def dump_unshared_checkpoint(self, unshared_mask):
        # return state_dict and do a deepcopy
        ret = dict()
        unshared_mask = set(unshared_mask)
        if unshared_mask:
            for k, v in self.state_dict().items():
                if k in unshared_mask:
                    ret[k] = v.clone().detach()
        return ret

    def load_unshared_checkpoint(self, unshared_mask, checkpoint, strict=True):
        to_load = dict()
        check_length = 0
        if unshared_mask:
            for k in unshared_mask:
                to_load[k] = checkpoint[k]
                check_length += 1
            if strict:
                assert check_length == len(checkpoint)
            self.load_state_dict(to_load, strict=False)
