""" Config class for search/augment """
import argparse
import os
from functools import partial

import yaml


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--config_file', type=str, default=None)
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', default=None, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--w_lr_scheduler', default='plateau', help='plateau / cosine')
        parser.add_argument('--w_lr', type=float, default=0.1, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.0001,
                            help='training will stop when learning rate drops to lower than this value')
        parser.add_argument('--w_lr_patience', type=int, default=2,
                            help='number of validations with no improvement after which learning rate will be reduced')
        parser.add_argument('--w_lr_factor', type=float, default=0.2,
                            help='factor by which the learning rate will be reduced')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--n_layers', type=int, default=8, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=0, help='# of workers')
        parser.add_argument('--search_all_alpha', action='store_true', default=False,
                            help='try all combinations of alphas (instead of the original darts way)')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')
        parser.add_argument('--n_nodes', type=int, default=4, help='number of nodes in one cell')
        parser.add_argument('--stem_multiplier', type=int, default=3, help='stem multiplier in cells')
        parser.add_argument('--shared_policy', type=str, default="all",
                            help='the rule to get shared weights')
        parser.add_argument('--shared_policy_kwargs', default=None,
                            help='the kwargs to get shared weights, use this in yml (as dict)')
        parser.add_argument('--validate_instance', default=1, type=int,
                            help='number of instances to validate, when searching all possible alpha combs')
        parser.add_argument('--validate_epochs', default=10, type=int,
                            help='run validation every n epochs')
        parser.add_argument('--cutout_length', default=0, type=int,
                            help='cutout length of data augmentation')
        parser.add_argument('--designated_subgraph', default=None,
                            help='designated subgraph (as list of indices), in config')
        parser.add_argument('--designated_training', default=None,
                            help='designated subgraph for training')
        parser.add_argument('--bn_momentum', default=0.4, help='batch norm momentum')
        parser.add_argument('--nni', default=None, type=str, help='mode, running with nni')
        parser.add_argument('--step_order', default=None, type=str,
                            help='the rule of order to update the subgraphs, write it with <rule>_<seed>')
        parser.add_argument('--dense_validation_steps', default=0, type=int,
                            help='for the last k steps, will do validation on all subgraphs every batch')
        parser.add_argument('--cutoff_epochs', default=None, type=int,
                            help='training will stop here (won\'t change learning rate decay, '
                                 'will affect dense validation)')

        # for finetuning experiments
        parser.add_argument('--save_weights_on_validation', default=False, action='store_true')
        parser.add_argument('--finetune_from_step', type=str, default=None,
                            help='path to pretrained checkpoint subdirectory for finetuning')
        parser.add_argument('--finetune_epochs', type=int, default=20,
                            help='another k epochs will be trained for finetuning')
        parser.add_argument('--finetune_max_steps', type=int, default=None,
                            help='will validate each step for max_steps')

        return parser

    def build_overriding_parser(self):
        parser = get_parser("Search config overriding")
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--w_lr', type=float, default=None)
        parser.add_argument('--w_gamma', type=float, default=None)
        parser.add_argument('--epochs', type=int, default=None)
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--step_order', type=str, default=None)
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        if self.config_file is not None:
            with open(self.config_file, "r") as fp:
                for k, v in yaml.safe_load(fp).items():
                    if not hasattr(self, k):
                        raise ValueError("Property {} not found in config".format(k))
                    setattr(self, k, v)

        overriding_parser = self.build_overriding_parser()
        args, _ = overriding_parser.parse_known_args()
        args = vars(args)
        for name in args:
            if args[name] is not None:
                setattr(self, name, args[name])

        if self.dataset is None:
            raise ValueError("Dataset config is required yet missing")
        if self.shared_policy_kwargs is None:
            self.shared_policy_kwargs = dict()
        self.data_path = './data/'
        self.path = os.path.join('outputs', self.name)
        self.plot_path = os.path.join(self.path, 'plots')
        self.checkpoint_path = os.path.join("./checkpoints/", self.name)
