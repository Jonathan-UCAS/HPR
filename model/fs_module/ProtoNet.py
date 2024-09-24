import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, k_way, n_shot, query):
        super().__init__()
        self.loss_fn = torch.nn.NLLLoss()
        self.k = k_way
        self.n = n_shot
        self.query = query

    def get_dist(self, feat):
        support = feat[:self.n * self.k]
        queries = feat[self.n * self.k:]
        prototype = support.reshape(self.k, self.n, -1).mean(1)
        distance = torch.cdist(queries.unsqueeze(0), prototype.unsqueeze(0)).squeeze(0)
        return distance

    def forward(self, feat, label):
        dist = self.get_dist(feat)
        y_pred = (-dist).softmax(1)
        log_p_y = (-dist).log_softmax(dim=1)
        loss = self.loss_fn(log_p_y, label[1].to(feat.device))
        return y_pred, loss


def start_tsne(x_train, y_train):
    print("initializing data visualization...")
    X_tsne = TSNE().fit_transform(x_train)
    plt.figure(figsize=(10, 10))
    # 为每个类的样本创建散点图
    # for class_value in range(len(y_train_final.unique())):
    for class_value in range(5):
        # 获取此类的示例的行索引
        row_ix = np.where(y_train == class_value)
        # 创建这些样本的散布
        plt.scatter(X_tsne[row_ix, 0], X_tsne[row_ix, 1], label=class_value, s=100)
    plt.legend(loc='best')
    # 绘制散点图
    plt.show()


if __name__ == '__main__':
    sample_inpt = torch.randn((20, 256))
    q_label = torch.arange(5).repeat_interleave(3)
    s_label = torch.arange(5)
    net = ProtoNet_Original(k_way=5, n_shot=1, query=3)
    pred, loss = net(sample_inpt, [s_label, q_label])
