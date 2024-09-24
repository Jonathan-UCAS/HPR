import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyptorch.nn import ToPoincare, FromPoincare
from hyptorch.pmath import poincare_mean, dist_matrix, dist0, dist, dist_matrix_knn, _mobius_add
from manifolds import Oblique

import geotorch
import geoopt
import geoopt_layers
from hyp_layers.Hyper_Conv import HNNLayer


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (55, 1024, 1024)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def knn_hyper(x, k):
    x = x.transpose(1, 2)  # (55,1024,3)
    e2p = ToPoincare(c=0.01, train_c=False, train_x=False)
    x = e2p(x)  # 55*1024*3
    pairwise_distance = dist_matrix_knn(x, x, c=e2p.c)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_hyper_graph_feature(x, k=20, idx=None):
    e2p = ToPoincare(c=0.01, train_c=False, train_x=False)
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    x = e2p(x)  # 55*1024*3
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((_mobius_add(feature, -x, c=e2p.c), x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class point_backbone(nn.Module):
    def __init__(self):
        super(point_backbone, self).__init__()
        self.k = 20
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),  # !!!! modify this by ymm
                                   self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.feat_transform = nn.Linear(1024, 256)
        self.bin_num = [1, 2, 4, 8, 16, 32]

    def forward(self, x):  # (B, 3, 1024)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        batch, feat_dim, n_points = x.shape
        bin_feat = []
        for bin in self.bin_num:
            z = x.view(batch, feat_dim, bin, -1)
            z_max, _ = z.max(3)
            z = z.mean(3) + z_max
            bin_feat.append(z)
        bin_feat = torch.cat(bin_feat, 2).permute(2, 0, 1).contiguous()  # 31/62, 6, 1024
        bin_feat = self.feat_transform(bin_feat)  # bins, batch, 1024
        return bin_feat


class DGCNN(nn.Module):
    def __init__(self, args=None, output_channels=100):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = 20
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.final_feat_dim = output_channels
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5, nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([30, 512, 1024])
        x = self.conv5(x)  # torch.Size([55, 1024, 1024])
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (55,1024)
        return x


class Ori_DGCNN(nn.Module):
    def __init__(self, args=None, output_channels=256):  # 原始100
        super(Ori_DGCNN, self).__init__()
        self.args = args
        self.k = 20
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.final_feat_dim = output_channels
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Hyper_DGCNN(nn.Module):
    def __init__(self, args=None, output_channels=256):  # 原始100
        super(Hyper_DGCNN, self).__init__()
        self.args = args
        self.k = 20
        self.c = 0.01
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn2_eu = nn.BatchNorm2d(64)
        self.bn3_eu = nn.BatchNorm2d(128)
        self.bn4_eu = nn.BatchNorm2d(256)
        self.bn5_eu = nn.BatchNorm1d(1024)
        poincare = geoopt.PoincareBall(c=self.c)
        self.conv1 = geoopt_layers.poincare.MobiusConv2d(dim=6, kernel_size=1, dim_out=64, ball=poincare)  # (B,P,D.W.H)
        self.bn1 = geoopt_layers.poincare.MobiusBatchNorm2d(64, ball=poincare, bias=False)
        self.hypAct1 = geoopt_layers.poincare.RadialNd(torch.nn.LeakyReLU(negative_slope=0.2), ball=poincare)

        self.conv2 = geoopt_layers.poincare.MobiusConv2d(dim=128, kernel_size=1, dim_out=64, ball=poincare)  # (B,P,D.W.H)
        self.bn2 = geoopt_layers.poincare.MobiusBatchNorm2d(64, ball=poincare, bias=False)
        self.hypAct2 = geoopt_layers.poincare.RadialNd(torch.nn.LeakyReLU(negative_slope=0.2), ball=poincare)

        self.conv3 = geoopt_layers.poincare.MobiusConv2d(dim=128, kernel_size=1, dim_out=128, ball=poincare)  # (B,P,D.W.H)
        self.bn3 = geoopt_layers.poincare.MobiusBatchNorm2d(128, ball=poincare, bias=False)
        self.hypAct3 = geoopt_layers.poincare.RadialNd(torch.nn.LeakyReLU(negative_slope=0.2), ball=poincare)

        self.conv4 = geoopt_layers.poincare.MobiusConv2d(dim=256, kernel_size=1, dim_out=256, ball=poincare)  # (B,P,D.W.H)
        self.bn4 = geoopt_layers.poincare.MobiusBatchNorm2d(256, ball=poincare, bias=False)
        self.hypAct4 = geoopt_layers.poincare.RadialNd(torch.nn.LeakyReLU(negative_slope=0.2), ball=poincare)

        self.conv5 = geoopt_layers.poincare.MobiusConv2d(dim=128, kernel_size=1, dim_out=256, ball=poincare)
        self.bn5 = geoopt_layers.poincare.MobiusBatchNorm1d(256, ball=poincare, bias=False)
        self.hypAct5 = geoopt_layers.poincare.RadialNd(torch.nn.LeakyReLU(negative_slope=0.2), ball=poincare)

        self.conv2_eu = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                      self.bn2_eu, nn.LeakyReLU(negative_slope=0.2))
        self.conv3_eu = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                      self.bn3_eu, nn.LeakyReLU(negative_slope=0.2))
        self.conv4_eu = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                      self.bn4_eu, nn.LeakyReLU(negative_slope=0.2))
        self.conv5_eu = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                      self.bn5_eu, nn.LeakyReLU(negative_slope=0.2))
        self.final_feat_dim = output_channels

    def forward(self, x):
        p2e = FromPoincare(c=self.c, train_c=False, train_x=False)
        batch_size = x.size(0)
        x = get_hyper_graph_feature(x, k=self.k)
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = self.bn1(x)
        x = self.hypAct1(x)

        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = x1.transpose(2, 1).contiguous()
        x1 = p2e(x1)  # back to Euclidean
        x1 = x1.transpose(2, 1).contiguous()

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2_eu(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3_eu(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4_eu(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (55,128,1024)

        x = self.conv5_eu(x)  # torch.Size([55, 1024, 1024])
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (55,1024)
        return x
