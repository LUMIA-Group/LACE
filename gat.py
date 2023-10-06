#!/usr/bin/env python
# -*- coding: utf-8 -*-
# dgl库介绍：https://docs.dgl.ai/tutorials/blitz/2_dglgraph.html
# GAT的dgl使用示例：https://docs.dgl.ai/tutorials/blitz/2_dglgraph.html （大意看这个。)
# GAT的dgl库标准实现：https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat （暂定使用这个。)
# GAT实现细节看这个：https://docs.dgl.ai/en/latest/guide/mixed_precision.html 。也是避免混合精度的问题。这个写的清晰易懂啊。更新：没有混合精度的问题了。
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import GATConv

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            # print('1 h是', h.dtype, h.float().dtype) # torch.float32 torch.float32
            # h = h.float()
            # print('2 h是', h.dtype, h.float().dtype)  # torch.float32 torch.float32
            h = self.gat_layers[l](self.g, h).flatten(1) # DGLError: The node features' data type torch.float16 doesn't match edge features' data type torch.float32, please convert them to the same type.
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits