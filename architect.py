""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import os
import random

import numpy as np
import torch

import genotypes
from visualize import plot


class Architect():
    """ Compute gradients of alphas """

    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y)  # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay * w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y)  # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi * h

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas())  # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas())  # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian


class SubgraphSearchOptimizer():

    @staticmethod
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

    def __init__(self, logger, config, net, optim):
        assert config.search_all_alpha
        self.logger = logger
        self.net = net
        self.optim = optim
        self.shuffle_rule, self.shuffle_state = None, None

        if config.step_order:
            self.shuffle_rule, step_seed = config.step_order.split("_")
            self.shuffle_state = np.random.RandomState(int(step_seed))
        self.all_archits = self.get_all_architecture_indices_combination(len(genotypes.PRIMITIVES),
                                                                         config.n_nodes * (config.n_nodes + 1) // 2)
        if config.designated_subgraph is not None:
            self.all_archits = [self.all_archits[t] for t in config.designated_subgraph]
        if self.shuffle_rule == "one":
            self.logger.info("Architecture order shuffled")
            self.shuffle_state.shuffle(self.all_archits)

        self.validate_instances = np.arange(len(self.all_archits))
        np.random.shuffle(self.validate_instances)
        self.validate_instances = sorted(self.validate_instances[:config.validate_instance])

        self.mapping = None

        # DISABLE MAPPING IN ARCHITECT FOR NOW
        # if config.shared_policy == "group":
        #     groups = config.shared_policy_kwargs["groups"]
        #     self.mapping = [i % groups for i in range(len(self.all_archits))]
        #     random.shuffle(self.mapping)
        #     logger.info("Group mapping: {}".format(self.mapping))

        self.unshared_mask = self.net.generate_unshared_mask(config.shared_policy, **config.shared_policy_kwargs)
        self.unshared_parameters = []
        for name, parameter in self.net.named_parameters():
            if name in self.unshared_mask:
                self.unshared_parameters.append(parameter)

        self.saved_weights = [None] * len(self.all_archits)
        self.saved_optimizer = [None] * len(self.all_archits)

        self.logger.info("Logging all generated architectures...")
        for i in range(len(self.all_archits)):
            self.save(i)
            archi = self.restore(i)

            genotype = self.net.genotype()
            self.logger.info("Index = {}, genotype = {}".format(i, genotype))

            plot_path = os.path.join(config.plot_path, "ALL{}".format(archi))
            caption = "#{:02d}: {}".format(i, archi)
            plot(genotype.normal, plot_path + "-normal", caption)

        self._cur_index = -1  # ready for beginning

        self.logger.info("Unsharing: {}".format(self.unshared_mask))
        self.logger.info("Possible combinations length: {}".format(len(self.all_archits)))
        self.net = net

    def __next__(self):
        # save, next, load
        if self._cur_index >= 0:
            self.save(self._cur_index)
        self._cur_index = (self._cur_index + 1) % len(self.all_archits)
        if self._cur_index == 0 and self.shuffle_rule == "every":
            self.logger.info("Architecture order shuffled")
            self.shuffle_state.shuffle(self.all_archits)
        return self.restore(self._cur_index)

    def __getitem__(self, item):
        # does not handle save
        return self.restore(item)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.all_archits)

    def dump_optimizer_state(self):
        ret = dict()
        if self.unshared_parameters:
            for parameter in self.unshared_parameters:
                if parameter in self.optim.state:
                    ret[parameter] = {k: v.clone().detach() for k, v in self.optim.state[parameter].items()}
        return ret

    def restore_optimizer_state(self, state):
        if state:
            self.optim.state.update(state)

    def save(self, index=None):
        if index is None:
            index = self._cur_index
        self.saved_weights[self.map(index)] = self.net.dump_unshared_checkpoint(self.unshared_mask)
        self.saved_optimizer[self.map(index)] = self.dump_optimizer_state()
        return index

    def restore(self, index):
        ckpt = self.saved_weights[self.map(index)]
        assert ckpt is not None, "Please save checkpoints before loading..."
        self.net.load_unshared_checkpoint(self.unshared_mask, ckpt)

        ckpt = self.saved_optimizer[self.map(index)]
        assert ckpt is not None
        self.restore_optimizer_state(ckpt)

        self.net.generate_and_fill_alphas("fixed", select=self.all_archits[index])
        return self.display(self.all_archits[index])

    @property
    def current_architecture(self):
        return self.display(self.all_archits[self._cur_index])

    def map(self, index):
        if self.mapping is None:
            return index
        return self.mapping[index]

    @staticmethod
    def display(t):
        return "".join(str(i + 1) for i in t)