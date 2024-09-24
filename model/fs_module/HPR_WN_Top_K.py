import torch
import numpy as np
import torch.nn as nn
from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix
from manifolds import Oblique
import torch.nn.functional as F
from util.dist import cos_sim
import geotorch


class HPR_WN_Top_K(nn.Module):
    def __init__(self, k_way, n_shot, query, c=0.02, train_c=False, train_x=False, temperature=1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = torch.nn.NLLLoss()

        self.k = k_way
        self.n = n_shot
        self.query = query
        self.temperature = temperature

    def get_dist(self, feat):
        support = feat[:self.n * self.k]  # 5*512
        queries = feat[self.n * self.k:]  # 50*512

        len_q = len(queries)
        support_ = support.reshape(self.k, self.n, -1)
        queries_ = queries.reshape(len_q, 1, -1)

        proto_e = support.reshape(self.k, self.n, -1).mean(1)  # 5*1*1024 ---> 5*512
        distance_e = torch.cdist(queries.unsqueeze(0), proto_e.unsqueeze(0)).squeeze(0)
        pred = torch.argmax(-distance_e, 1)
        s0 = support_[0]
        s1 = support_[1]
        s2 = support_[2]
        s3 = support_[3]
        s4 = support_[4]
        for i in range(len_q):
            if pred[i] == 0:
                s0 = torch.cat((s0, queries_[i]), 0)
            elif pred[i] == 1:
                s1 = torch.cat((s1, queries_[i]), 0)
            elif pred[i] == 2:
                s2 = torch.cat((s2, queries_[i]), 0)
            elif pred[i] == 3:
                s3 = torch.cat((s3, queries_[i]), 0)
            else:
                s4 = torch.cat((s4, queries_[i]), 0)

        s0_new = support_[0]
        s1_new = support_[1]
        s2_new = support_[2]
        s3_new = support_[3]
        s4_new = support_[4]

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

        for i in range(70):
            if s0.shape[0] > 1:
                s0_new = torch.cat((s0_new, torch.normal(mean=s0_, std=s0_std)), dim=0)  # 101*512
            if s1.shape[0] > 1:
                s1_new = torch.cat((s1_new, torch.normal(mean=s1_, std=s1_std)), dim=0)
            if s2.shape[0] > 1:
                s2_new = torch.cat((s2_new, torch.normal(mean=s2_, std=s2_std)), dim=0)
            if s3.shape[0] > 1:
                s3_new = torch.cat((s3_new, torch.normal(mean=s3_, std=s3_std)), dim=0)
            if s4.shape[0] > 1:
                s4_new = torch.cat((s4_new, torch.normal(mean=s4_, std=s4_std)), dim=0)

        if s0_new.shape[0] > 1:
            distance_new0 = -torch.cdist(s0_new.unsqueeze(0), proto_e[0].reshape(1, -1).unsqueeze(0)).squeeze(0)  # 101*1
            _, indices_0 = distance_new0.topk(50, dim=0, largest=True, sorted=True)
            s0_new = s0_new[indices_0.squeeze(1)]  # 本来是20 top 8-->20-->15-->120 select 100
        if s1_new.shape[0] > 1:
            distance_new1 = -torch.cdist(s1_new.unsqueeze(0), proto_e[1].reshape(1, -1).unsqueeze(0)).squeeze(0)
            _, indices_1 = distance_new1.topk(50, dim=0, largest=True, sorted=True)
            s1_new = s1_new[indices_1.squeeze(1)]
        if s2_new.shape[0] > 1:
            distance_new2 = -torch.cdist(s2_new.unsqueeze(0), proto_e[2].reshape(1, -1).unsqueeze(0)).squeeze(0)
            _, indices_2 = distance_new2.topk(50, dim=0, largest=True, sorted=True)
            s2_new = s2_new[indices_2.squeeze(1)]
        if s3_new.shape[0] > 1:
            distance_new3 = -torch.cdist(s3_new.unsqueeze(0), proto_e[3].reshape(1, -1).unsqueeze(0)).squeeze(0)
            _, indices_3 = distance_new3.topk(50, dim=0, largest=True, sorted=True)
            s3_new = s3_new[indices_3.squeeze(1)]
        if s4_new.shape[0] > 1:
            distance_new4 = -torch.cdist(s4_new.unsqueeze(0), proto_e[4].reshape(1, -1).unsqueeze(0)).squeeze(0)
            _, indices_4 = distance_new4.topk(50, dim=0, largest=True, sorted=True)
            s4_new = s4_new[indices_4.squeeze(1)]

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

        y_pred = (-dist_new).softmax(1)
        # y_pred_ = (-dist_).softmax(1)
        log_p_y = (-dist).log_softmax(dim=1)

        loss = self.loss_fn(log_p_y, label[1].to(feat.device))
        loss = loss

        return y_pred, loss


if __name__ == '__main__':
    sample_inpt = torch.randn((20, 256))
    q_label = torch.arange(5).repeat_interleave(3)
    s_label = torch.arange(5)
    net = HPR_WN_Top_K(k_way=5, n_shot=1, query=3)
    pred, loss = net(sample_inpt, [s_label, q_label])
