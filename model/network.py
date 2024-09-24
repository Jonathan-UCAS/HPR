import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======= backbone ============
from backbone.point_base import point_backbone, DGCNN, Ori_DGCNN, Hyper_DGCNN
# =============================


# ======== fs algorithm =========
from fs_module.CIA import CIA
from fs_module.MetaOpt import MetaOpt
from fs_module.RelationNet import RelationNetwork
from fs_module.crossAtt_mixmodel import crossAtt_mixmodel
from fs_module.ProtoNet import ProtoNet
from fs_module.PR_N import PR_N
# the main methods we proposed: HPR-D, HPR-WN and Top-K HPR-WN
from fs_module.HPR_D import HPR_D
from fs_module.HPR_WN import HPR_WN
from fs_module.HPR_WN_Top_K import HPR_WN_Top_K
# ===============================
from util.utils import init_weights


class fs_network(nn.Module):
    def __init__(self, k_way, n_shot, query, backbone='dgcnn_normal', fs='wrapped'):
        super().__init__()
        self.k = k_way
        self.n = n_shot
        self.query = query
        self.label = None
        self.backbone = backbone
        if backbone in ['dgcnn']:
            self.pointbb = self.get_backbone(backbone)
        elif backbone in 'dgcnn_normal':
            self.pointbb = self.get_backbone(backbone)
        elif backbone in 'ori_dgcnn':
            self.pointbb = self.get_backbone(backbone)
        elif backbone in 'hyper_dgcnn':
            self.pointbb = self.get_backbone(backbone)
        self.fs_head = self.get_fs_head(fs)
        self.fs = fs

    def get_fs_head(self, fs):
        if fs == 'crossAtt_mixmodel':
            return crossAtt_mixmodel(self.k, self.n, self.query)
        elif fs == 'CIA':
            return CIA(self.k, self.n, self.query)
        elif fs == 'ori_protonet':
            return ProtoNet(self.k, self.n, self.query)
        elif fs == 'protonet':
            return PR_N(self.k, self.n, self.query)
        elif fs == 'hyperproto':
            return HPR_D(self.k, self.n, self.query)
        elif fs == 'wrapped':
            return HPR_WN(self.k, self.n, self.query)
        elif fs == 'protonet_top':
            return HPR_WN_Top_K(self.k, self.n, self.query)
        elif fs == 'MetaOpt':
            return MetaOpt()
        elif fs == 'relationnet':
            return RelationNetwork(256, 256)
        else:
            raise ValueError('Illegal fs_head')

    def get_backbone(self, backbone):
        if backbone == 'dgcnn':
            return point_backbone()
        elif backbone == 'dgcnn_normal':
            return DGCNN()
        elif backbone == 'ori_dgcnn':
            return Ori_DGCNN()
        elif backbone == 'hyper_dgcnn':
            return Hyper_DGCNN()
        else:
            raise ValueError('Illegal Backbone')

    def forward(self, x, label, Istest=False):
        self.label = label
        if self.backbone == 'pointAndview':
            point_embed = self.pointbb(x)
            view_embed = self.viewbb(x)
            embeding = torch.cat([point_embed, view_embed])
            # embeding = point_embed
        elif self.backbone in ['dgcnn']:
            embeding = self.pointbb(x)  # (20,256)
        elif self.backbone in ['dgcnn_normal']:
            embeding = self.pointbb(x)  # (20,256)
        elif self.backbone in ['ori_dgcnn']:
            embeding = self.pointbb(x)  # (20,256)
        elif self.backbone in ['hyper_dgcnn']:
            embeding = self.pointbb(x)  # (20,256)
        elif self.backbone == 'view':
            embeding = self.viewbb(x)
        if self.fs == 'MetaOpt':
            support, query = embeding[:self.k * self.n, :], embeding[self.k * self.n:, :]
            support = support.reshape(1, self.k * self.n, -1)
            query = query.reshape(1, self.k * self.query, -1)
            support_labels, query_labels = torch.cuda.LongTensor(label[0]), torch.cuda.LongTensor(label[1])
            support_labels = support_labels.reshape(1, -1)
            # query_labels = 
            pred, loss = self.fs_head(query, support, support_labels, query_labels, self.k, self.n)
            return pred, loss
        if self.fs == 'protonet':
            # support_labels, query_labels = torch.cuda.LongTensor(label[0]), torch.cuda.LongTensor(label[1])
            # support_labels = support_labels.reshape(1, -1)
            pred, loss = self.fs_head(embeding, label)
            return pred, loss
        if self.fs == 'ori_protonet':
            # support_labels, query_labels = torch.cuda.LongTensor(label[0]), torch.cuda.LongTensor(label[1])
            # support_labels = support_labels.reshape(1, -1)
            pred, loss = self.fs_head(embeding, label)
            return pred, loss
        if self.fs == 'relationnet':
            # support_labels, query_labels = torch.cuda.LongTensor(label[0]), torch.cuda.LongTensor(label[1])
            # support_labels = support_labels.reshape(1, -1)
            pred, loss = self.fs_head(embeding, label)
            return pred, loss
        if self.fs == 'protonet_top':
            # support_labels, query_labels = torch.cuda.LongTensor(label[0]), torch.cuda.LongTensor(label[1])
            # support_labels = support_labels.reshape(1, -1)
            pred, loss = self.fs_head(embeding, label)
            return pred, loss
        if self.fs == 'hyperproto':
            # support_labels, query_labels = torch.cuda.LongTensor(label[0]), torch.cuda.LongTensor(label[1])
            # support_labels = support_labels.reshape(1, -1)
            pred, loss = self.fs_head(embeding, label)
            return pred, loss
        if self.fs == 'wrapped':
            # support_labels, query_labels = torch.cuda.LongTensor(label[0]), torch.cuda.LongTensor(label[1])
            # support_labels = support_labels.reshape(1, -1)
            pred, loss = self.fs_head(embeding, label)
            return pred, loss
        elif self.fs in ['trip_text', 'mixmodel']:
            pred, loss = self.fs_head(point_embed, view_embed, label)
        elif self.fs in ['CIA', 'crossAtt_mixmodel']:
            pred, loss = self.fs_head(embeding, label)
        else:
            raise Exception("Not implemented error")

        return pred, loss


if __name__ == '__main__':
    fs_net = fs_network(k_way=5, n_shot=1, query=3).cuda()
    sample_inpt = torch.randn((20, 3, 1024)).cuda()
    pred, loss = fs_net(sample_inpt)
    a = 1
