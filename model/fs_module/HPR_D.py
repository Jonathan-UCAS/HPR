import torch
import numpy as np
import torch.nn as nn
from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix, dist0, dist
from manifolds import Oblique
import torch.nn.functional as F
from util.dist import cos_sim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import geotorch


class HPR_D(nn.Module):
    def __init__(self, k_way, n_shot, query, c=0.01, train_c=False, train_x=False, temperature=1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.k = k_way
        self.n = n_shot
        self.query = query
        self.c0 = torch.randn(1).cuda()
        self.temperature = temperature

    def get_dist(self, feat):
        self.e2p = ToPoincare(c=0.01, train_c=False, train_x=False)  # 5shot 采用0.005,1shot 采用0.01

        support = feat[:self.n * self.k]  # 25*512
        queries = feat[self.n * self.k:]  # 50*512]

        support_p = self.e2p(support)
        queries_p = self.e2p(queries)  # 50*512
        support_p_ = support_p.reshape(self.k, self.n, -1)  # 5*5*512
        proto_p = poincare_mean(support_p_, dim=1, c=self.e2p.c)  # 5*512

        len_q = len(queries_p)
        queries_p_ = queries_p.reshape(len_q, 1, -1)  # 50*1*512

        distance_p = dist_matrix(queries_p, proto_p, c=self.e2p.c) / self.temperature  # 50*5
        pred = torch.argmax(-distance_p, 1)

        s0, s1, s2, s3, s4 = support_p_

        for i in range(len_q):
            if pred[i] == 0:
                s0 = torch.cat((s0, queries_p_[i]), 0)
            elif pred[i] == 1:
                s1 = torch.cat((s1, queries_p_[i]), 0)
            elif pred[i] == 2:
                s2 = torch.cat((s2, queries_p_[i]), 0)
            elif pred[i] == 3:
                s3 = torch.cat((s3, queries_p_[i]), 0)
            else:
                s4 = torch.cat((s4, queries_p_[i]), 0)

        s0_new = poincare_mean(s0, dim=0, c=self.e2p.c).reshape(1, -1)  # 1*512
        s1_new = poincare_mean(s1, dim=0, c=self.e2p.c).reshape(1, -1)
        s2_new = poincare_mean(s2, dim=0, c=self.e2p.c).reshape(1, -1)
        s3_new = poincare_mean(s3, dim=0, c=self.e2p.c).reshape(1, -1)
        s4_new = poincare_mean(s4, dim=0, c=self.e2p.c).reshape(1, -1)

        proto_new = torch.cat((s0_new, s1_new, s2_new, s3_new, s4_new), 0)  # 5*512
        distance_p_new = dist_matrix(queries_p, proto_new, c=self.e2p.c) / self.temperature  # 50*5

        return distance_p, distance_p_new

    def forward(self, feat, label):
        dist, dist_new = self.get_dist(feat)
        y_pred = (-dist_new).softmax(1)  # dist or dist_new
        log_p_y = (-dist).log_softmax(dim=1)
        loss = self.loss_fn(log_p_y, label[1].to(feat.device))
        return y_pred, loss

    def find_min(self, s):
        anchor = dist0(s[0], c=self.e2p.c)
        index = 0
        for j in range(len(s) - 1):
            if anchor > dist0(s[j + 1], c=self.e2p.c):
                anchor = dist0(s[j + 1], c=self.e2p.c)
                index = j + 1
        return anchor, index

    def delta_hyp(self, dismat):
        """
        computes delta hyperbolicity value from distance matrix
        """
        p = 0
        row = torch.unsqueeze(dismat[p, :], 0)
        col = torch.unsqueeze(dismat[:, p], 1)
        XY_p = 0.5 * (row + col - dismat)
        maxmin = torch.max(torch.minimum(XY_p[:, :, None], XY_p[None, :, :]), 1)[0]  # A*A
        # maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)  # A*A
        return torch.max(maxmin - XY_p)  # A*A-A

    def batch_delta_hyp(self, X):
        distmat = self.distance_matrix(X, X)
        diam = torch.max(distmat)
        delta_rel = 2 * self.delta_hyp(distmat) / diam
        # vals.append(delta_rel)
        return delta_rel

    def distance_matrix(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = ((a - b) ** 2).sum(dim=2)
        logits = torch.sqrt(logits)
        return logits


def start_tsne(x_train, y_train):
    print("正在进行初始输入数据的可视化...")
    X_tsne = TSNE().fit_transform(x_train)
    plt.figure(figsize=(10, 10))
    for class_value in range(5):
        row_ix = np.where(y_train == class_value)
        plt.scatter(X_tsne[row_ix, 0], X_tsne[row_ix, 1], label=class_value)

    plt.legend(loc='best')
    # 绘制散点图
    plt.show()


if __name__ == '__main__':
    sample_inpt = torch.randn((20, 256))
    q_label = torch.arange(5).repeat_interleave(3)
    s_label = torch.arange(5)
    net = HPR_D(k_way=5, n_shot=1, query=3)
    pred, loss = net(sample_inpt, [s_label, q_label])
