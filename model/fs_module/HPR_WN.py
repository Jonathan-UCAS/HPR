import torch
import numpy as np
import torch.nn as nn
from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix, dist, ptransp0_p, expmap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from manifolds import Oblique
import torch.nn.functional as F
from util.dist import cos_sim
import geotorch


class HPR_WN(nn.Module):
    def __init__(self, k_way, n_shot, query, c=0.005, train_c=False, train_x=False, temperature=1):  # c=0.02为最佳
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = torch.nn.NLLLoss()

        self.k = k_way
        self.n = n_shot
        self.query = query
        self.c0 = torch.randn(1).cuda()
        self.linear_c = nn.Linear(1, 1)
        self.temperature = temperature

        self.cosine_w = torch.randn(1).cuda()
        self.linear_cw = nn.Linear(1, 1)

        self.dis_w = torch.randn(1).cuda()
        self.linear_dw = nn.Linear(1, 1)

    def get_dist(self, feat):
        # c = torch.sigmoid(self.linear_c(self.c0))
        self.e2p = ToPoincare(c=0.01, train_c=False, train_x=False)  # c settings. 5-shot: 0.005, 1-shot: 0.01.
        c1 = self.e2p.c

        support = feat[:self.n * self.k]  # 25*512
        support_e = support.reshape(self.k, self.n, -1)  # 5*5*512
        queries = feat[self.n * self.k:]  # 50*512
        queries_e = queries.unsqueeze(1)  # 50*1*512

        support_p = self.e2p(support)
        queries_p = self.e2p(queries)  # 50*512

        support_p_ = support_p.reshape(self.k, self.n, -1)  # [K, N, C]
        proto_p = poincare_mean(support_p_, dim=1, c=self.e2p.c)  # [K, C]

        """
        norm_p = F.normalize(proto_p, p=2, dim=1, eps=1e-12)  # 5*1024
        norm_q = F.normalize(queries, p=2, dim=1, eps=1e-12)  # 50*1024
        norm_p = norm_p.transpose(0, 1)
        sims = torch.matmul(norm_q, norm_p)
        std = torch.std(sims)  # 约束类间余弦方差损失，尽可能大，后续增加类内方差约束损失，越小越好
        思考一下双曲空间中的余弦形式
        """
        len_q = len(queries_p)
        queries_p_ = queries_p.reshape(len_q, 1, -1)  # 50*1*512

        distance_p = dist_matrix(queries_p, proto_p, c=self.e2p.c) / self.temperature  # 50*5
        pred = torch.argmax(-distance_p, 1)

        s0_e = support_e[0]  # 5*512,计算欧式空间的方差
        s1_e = support_e[1]
        s2_e = support_e[2]
        s3_e = support_e[3]
        s4_e = support_e[4]

        s0 = support_p_[0]  # 5*512
        s1 = support_p_[1]
        s2 = support_p_[2]
        s3 = support_p_[3]
        s4 = support_p_[4]

        for i in range(len_q):  # 伪标签
            if pred[i] == 0:
                s0 = torch.cat((s0, queries_p_[i]), 0)  # n*512
                s0_e = torch.cat((s0_e, queries_e[i]), 0)
            elif pred[i] == 1:
                s1 = torch.cat((s1, queries_p_[i]), 0)
                s1_e = torch.cat((s1_e, queries_e[i]), 0)
            elif pred[i] == 2:
                s2 = torch.cat((s2, queries_p_[i]), 0)
                s2_e = torch.cat((s2_e, queries_e[i]), 0)
            elif pred[i] == 3:
                s3 = torch.cat((s3, queries_p_[i]), 0)
                s3_e = torch.cat((s3_e, queries_e[i]), 0)
            else:
                s4 = torch.cat((s4, queries_p_[i]), 0)
                s4_e = torch.cat((s4_e, queries_e[i]), 0)

        s0_p = s0.reshape(1, -1, 1024)
        s1_p = s1.reshape(1, -1, 1024)
        s2_p = s2.reshape(1, -1, 1024)
        s3_p = s3.reshape(1, -1, 1024)
        s4_p = s4.reshape(1, -1, 1024)

        s0_new = support_p_[0]  # 5*512
        s1_new = support_p_[1]
        s2_new = support_p_[2]
        s3_new = support_p_[3]
        s4_new = support_p_[4]

        s0_p = poincare_mean(s0_p, dim=1, c=self.e2p.c)  # 1*512
        s0_std = torch.std(s0_e, dim=0).reshape(1, -1)  # 欧式空间的方差作为包裹分布的方差

        s1_p = poincare_mean(s1_p, dim=1, c=self.e2p.c)
        s1_std = torch.std(s1_e, dim=0).reshape(1, -1)  # 1*512

        s2_p = poincare_mean(s2_p, dim=1, c=self.e2p.c)
        s2_std = torch.std(s2_e, dim=0).reshape(1, -1)  # 1*512

        s3_p = poincare_mean(s3_p, dim=1, c=self.e2p.c)
        s3_std = torch.std(s3_e, dim=0).reshape(1, -1)  # 1*512

        s4_p = poincare_mean(s4_p, dim=1, c=self.e2p.c)
        s4_std = torch.std(s4_e, dim=0).reshape(1, -1)  # 1*512

        for i in range(40):
            if s0.shape[0] > 1:
                s0_v = torch.normal(mean=0, std=s0_std)
                s0_pt_v = ptransp0_p(s0_p, s0_v, c=c1)  # 将s0_v从原点平行传输到均值s0_p点处得到s0_pt_v
                s0_exp = expmap(s0_p, s0_pt_v, c=c1)  #
                s0_new = torch.cat((s0_new, s0_exp), dim=0)
            if s1.shape[0] > 1:
                s1_v = torch.normal(mean=0, std=s1_std)
                s1_pt_v = ptransp0_p(s1_p, s1_v, c=c1)
                s1_exp = expmap(s1_p, s1_pt_v, c=c1)
                s1_new = torch.cat((s1_new, s1_exp), dim=0)
            if s2.shape[0] > 1:
                s2_v = torch.normal(mean=0, std=s2_std)
                s2_pt_v = ptransp0_p(s2_p, s2_v, c=c1)
                s2_exp = expmap(s2_p, s2_pt_v, c=c1)
                s2_new = torch.cat((s2_new, s2_exp), dim=0)
            if s3.shape[0] > 1:
                s3_v = torch.normal(mean=0, std=s3_std)
                s3_pt_v = ptransp0_p(s3_p, s3_v, c=c1)
                s3_exp = expmap(s3_p, s3_pt_v, c=c1)
                s3_new = torch.cat((s3_new, s3_exp), dim=0)
            if s4.shape[0] > 1:
                s4_v = torch.normal(mean=0, std=s4_std)
                s4_pt_v = ptransp0_p(s4_p, s4_v, c=c1)
                s4_exp = expmap(s4_p, s4_pt_v, c=c1)
                s4_new = torch.cat((s4_new, s4_exp), dim=0)

        s0_5 = torch.cat((s0_new, s1_new, s2_new, s3_new, s4_new), 0)
        # delta_re = self.batch_delta_hyp(s0_5)
        # print("delta_re为：",delta_re)
        # 将伪标签和新生成样本concat，比例为近似10:10
        """
        s0_new = torch.cat((s0, s0_new), dim=0)
        s1_new = torch.cat((s1, s1_new), dim=0)
        s2_new = torch.cat((s2, s2_new), dim=0)
        s3_new = torch.cat((s3, s3_new), dim=0)
        s4_new = torch.cat((s4, s4_new), dim=0)
        """

        s0_new = poincare_mean(s0_new, dim=0, c=self.e2p.c).reshape(1, -1)
        s1_new = poincare_mean(s1_new, dim=0, c=self.e2p.c).reshape(1, -1)
        s2_new = poincare_mean(s2_new, dim=0, c=self.e2p.c).reshape(1, -1)
        s3_new = poincare_mean(s3_new, dim=0, c=self.e2p.c).reshape(1, -1)
        s4_new = poincare_mean(s4_new, dim=0, c=self.e2p.c).reshape(1, -1)

        proto_new = torch.cat((s0_new, s1_new, s2_new, s3_new, s4_new), 0)  # 5*512
        distance_p_new = dist_matrix(queries_p, proto_new, c=self.e2p.c) / self.temperature  # 50*5

        return distance_p, distance_p_new

    def forward(self, feat, label):

        # # 可视化
        # global_transfeat = feat[5:].cpu().numpy()
        # global_fea = np.array(global_transfeat)
        # label_ = np.array(label[1].cpu().numpy())
        # start_tsne(global_fea, label_)

        dist, dist_new = self.get_dist(feat)

        y_pred = (-dist_new).softmax(1)  # dist or dist_new
        log_p_y = (-dist).log_softmax(dim=1)

        loss = self.loss_fn(log_p_y, label[1].to(feat.device))

        return y_pred, loss

    def delta_hyp(self, dismat):
        """
        computes delta hyperbolicity value from distance matrix
        """

        p = 0
        row = torch.unsqueeze(dismat[p, :], 0)
        col = torch.unsqueeze(dismat[:, p], 1)
        # row = dismat[p, :][np.newaxis, :]
        # col = dismat[:, p][:, np.newaxis]
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
        m = b.shape[0]  # unsqueeze先对指定维度进行升维，但升的维度只能为1，expand只能对维度为1的维度进行广播，复制
        a = a.unsqueeze(1).expand(n, m, -1)
        # print("a的shape",a.shape)
        b = b.unsqueeze(0).expand(n, m, -1)

        logits = ((a - b) ** 2).sum(dim=2)
        # print('logits',logits.shape)  # nan
        logits = torch.sqrt(logits)
        return logits


def start_tsne(x_train, y_train):
    print("visualizing...")
    X_tsne = TSNE().fit_transform(x_train)
    plt.figure(figsize=(10, 10))
    for class_value in range(5):
        row_ix = np.where(y_train == class_value)
        plt.scatter(X_tsne[row_ix, 0], X_tsne[row_ix, 1], label=class_value)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    sample_inpt = torch.randn((75, 512))
    q_label = torch.arange(5).repeat_interleave(3)
    s_label = torch.arange(5)
    net = wrappedNormal(k_way=5, n_shot=5, query=3)
    pred, loss = net(sample_inpt, [s_label, q_label])
