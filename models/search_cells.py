""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import ops


class SearchCell(nn.Module):
    """
    Cell for search
    Each edge is mixed and continuous relaxed.

    The cell is also simplified.
    """

    def __init__(self, n_nodes, C_p, C, bn_momentum):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_p : C_out[k-1]
            C   : C_in[k] (current)
        """
        super().__init__()
        self.n_nodes = n_nodes

        self.preproc = ops.StdConv(C_p, C, 1, 1, 0, affine=False, bn_momentum=bn_momentum)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(1 + i):  # include 1 input node
                op = ops.MixedOp(C, stride=1, bn_momentum=bn_momentum)
                self.dag[i].append(op)

    def forward(self, s, w_dag):
        s = self.preproc(s)

        states = [s]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)

        s_out = torch.cat(states[1:], dim=1)
        return s_out
