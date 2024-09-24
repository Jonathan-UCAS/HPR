import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
# from torch_scatter import scatter, scatter_add
# from torch_geometric.nn.conv import MessagePassing
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math
# from torch_geometric.nn.inits import glorot, zeros
from hyptorch.pmath import mobius_matvec, logmap0, project, proj_tan0, mobius_add, expmap0


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_features, out_features, c, dropout, act=F.leaky_relu, use_bias=False):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h

class HypLinear(nn.Module):
    """
    Poincare linear layer.
    """
    def __init__(self, in_features, out_features, c, dropout=0.6, use_bias=False):
        super(HypLinear, self).__init__()
        # self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    # def reset_parameters(self):
    #     glorot(self.weight)
    #     zeros(self.bias)

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = mobius_matvec(drop_weight, x, c=self.c)
        res = project(mv, c=self.c)
        if self.use_bias:
            bias = proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = expmap0(bias, c=self.c)
            hyp_bias = project(hyp_bias, c=self.c)
            res = mobius_add(res, hyp_bias, c=self.c)
            res = project(res, c=self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )



class HypAct(Module):
    """
    Poincare activation layer.
    """
    def __init__(self, c_in, c_out, act=F.leaky_relu):
        super(HypAct, self).__init__()
        # self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(logmap0(x, c=self.c_in))
        xt = proj_tan0(xt, c=self.c_out)
        return project(expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


# class HypAgg(MessagePassing):
#     """
#     Poincare aggregation layer using degree.
#     """
#     def __init__(self, manifold, c, out_features, device, bias=True):
#         super(HypAgg, self).__init__()
#         self.manifold = manifold
#         self.c = c
#         self.device = device
#         self.use_bias = bias
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features).to(device))
#         else:
#             self.register_parameter('bias', None)
#         zeros(self.bias)
#         self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1).to(device))
#
#     @staticmethod
#     def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
#         if edge_weight is None:
#             edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                      device=edge_index.device)
#
#         fill_value = 1 if not improved else 2
#         edge_index, edge_weight = add_remaining_self_loops(
#             edge_index, edge_weight, fill_value, num_nodes)
#
#         row, col = edge_index
#         deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#
#         return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#
#     def forward(self, x, edge_index=None):
#         x_tangent = self.manifold.logmap0(x, c=self.c)
#         edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
#         node_i = edge_index[0]
#         node_j = edge_index[1]
#         x_j = torch.nn.functional.embedding(node_j, x_tangent)
#         support = norm.view(-1, 1) * x_j
#         support_t = scatter(support, node_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
#         output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
#         return output
#
#     def extra_repr(self):
#         return 'c={}'.format(self.c)


