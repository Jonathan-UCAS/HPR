import torch
import numpy as np
import torch.nn as nn
from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix
from manifolds import Oblique
import torch.nn.functional as F
from util.dist import cos_sim
import geotorch


class PR_N(nn.Module):
    def __init__(self, k_way, n_shot, query, c=0.02, train_c=False, train_x=False, temperature=1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.k = k_way
        self.n = n_shot
        self.query = query

    def get_dist(self, feat):
        support = feat[:self.n * self.k]  # 5*512
        queries = feat[self.n * self.k:]  # 50*512
        len_q = len(queries)
        support_ = support.reshape(self.k, self.n, -1)
        queries_ = queries.reshape(len_q, 1, -1)

        proto_e = support.reshape(self.k, self.n, -1).mean(1)  # 5*1*512 ---> 5*512
        distance_e = torch.cdist(queries.unsqueeze(0), proto_e.unsqueeze(0)).squeeze(0)
        pred = torch.argmax(-distance_e, 1)

        s0, s1, s2, s3, s4 = support_

        support_dict = {0: s0, 1: s1, 2: s2, 3: s3, 4: s4}

        for i in range(len(pred)):
            support_dict[pred[i]] = torch.cat((support_dict[pred[i]], queries_[i]), 0)

        s0, s1, s2, s3, s4 = support_dict.values()

        s0_new, s1_new, s2_new, s3_new, s4_new = support_

        s0_ = s0.mean(0).reshape(1, -1)  # 1*512
        s0_std = torch.std(s0, dim=0).reshape(1, -1)  # 1*512
        s1_ = s1.mean(0).reshape(1, -1)
        s1_std = torch.std(s1, dim=0).reshape(1, -1)  # 1*512
        s2_ = s2.mean(0).reshape(1, -1)
        s2_std = torch.std(s2, dim=0).reshape(1, -1)  # 1*512
        s3_ = s3.mean(0).reshape(1, -1)
        s3_std = torch.std(s3, dim=0).reshape(1, -1)  # 1*512
        s4_ = s4.mean(0).reshape(1, -1)
        s4_std = torch.std(s4, dim=0).reshape(1, -1)  # 1*512

        for _ in range(50):
            if s0.shape[0] > 1:
                s0_new = torch.cat((s0_new, torch.normal(mean=s0_, std=s0_std)), dim=0)
            if s1.shape[0] > 1:
                s1_new = torch.cat((s1_new, torch.normal(mean=s1_, std=s1_std)), dim=0)
            if s2.shape[0] > 1:
                s2_new = torch.cat((s2_new, torch.normal(mean=s2_, std=s2_std)), dim=0)
            if s3.shape[0] > 1:
                s3_new = torch.cat((s3_new, torch.normal(mean=s3_, std=s3_std)), dim=0)
            if s4.shape[0] > 1:
                s4_new = torch.cat((s4_new, torch.normal(mean=s4_, std=s4_std)), dim=0)

        s0_new = s0_new.mean(0).reshape(1, -1)
        s1_new = s1_new.mean(0).reshape(1, -1)
        s2_new = s2_new.mean(0).reshape(1, -1)
        s3_new = s3_new.mean(0).reshape(1, -1)
        s4_new = s4_new.mean(0).reshape(1, -1)

        proto_new = torch.cat((s0_new, s1_new, s2_new, s3_new, s4_new), 0)
        distance_new = torch.cdist(queries.unsqueeze(0), proto_new.unsqueeze(0)).squeeze(0)

        return distance_e, distance_new

    def forward(self, feat, label):
        dist, dist_new = self.get_dist(feat)
        y_pred = (-dist_new).softmax(1)  # dist or dist_new
        log_p_y = (-dist).log_softmax(dim=1)
        loss = self.loss_fn(log_p_y, label[1].to(feat.device))
        return y_pred, loss


if __name__ == '__main__':
    sample_inpt = torch.randn((20, 256))
    q_label = torch.arange(5).repeat_interleave(3)
    s_label = torch.arange(5)
    net = ProtoNet(k_way=5, n_shot=1, query=3)
    pred, loss = net(sample_inpt, [s_label, q_label])
