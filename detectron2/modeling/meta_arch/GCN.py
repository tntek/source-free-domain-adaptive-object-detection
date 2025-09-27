import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pdb
# from torch_geometric.nn import GCNConv
from math import sqrt

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            #print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            #print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            #print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Feat2Graph(nn.Module):
    def __init__(self, num_feats):
        super(Feat2Graph, self).__init__()
        self.wq = nn.Linear(num_feats, num_feats)
        self.wk = nn.Linear(num_feats, num_feats)

    def forward(self, x):
        qx = self.wq(x)
        kx = self.wk(x)

        dot_mat = qx.matmul(kx.transpose(-1, -2))
        adj = F.normalize(dot_mat.square(), p=1, dim=-1)
        return x, adj

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=False, init="xavier"):
        super(GCN, self).__init__()
        self.graph = Feat2Graph(nfeat)

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nhid, init=init)
        self.gc3 = GraphConvolution(nhid, nfeat, init=init)
        self.dropout = dropout


    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x):
        x_in = x

        x, adj = self.graph(x)
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))

        return x
import torchvision

class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.graph = Feat2Graph(nfeat)

        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc_pol = nn.Linear(2048,1024)
        self.bn = torchvision.ops.FrozenBatchNorm2d(num_features=1024)
        self.gn = nn.GroupNorm(num_groups=32, num_channels=1024)
    # def bottleneck(self, path1, path2, path3, adj, in_x):
    #     return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x):

        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        x = F.relu(self.bn(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc_pol(x)

        return x
    
class AttentionWithSeparation(nn.Module):
    def __init__(self, dim=2048, dropout=0.1):
        super(AttentionWithSeparation, self).__init__()
        # Linear layers for query, key, and value mappings
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        # Two separate output layers: one for clustering, one for contrastive learning
        # self.out_clustering = nn.Linear(dim, dim)  # 聚类任务的输出层
        # self.out_contrastive = nn.Linear(dim, dim)  # 对比学习任务的输出层
        
        # # Dropout layer to avoid overfitting
        # self.dropout = nn.Dropout(dropout)

    def forward(self, strong_enhanced, weak_enhanced):
        """
        strong_enhanced: 强增强特征 [B, D]
        weak_enhanced: 弱增强特征 [B, D]
        """
        # Step 1: 拼接特征
        concatenated_features = torch.cat((strong_enhanced, weak_enhanced), dim=-1)  # [B, 2D]

        # Step 2: 计算 attention
        q = self.q(concatenated_features).unsqueeze(1)  # [B, 1, D]
        k = self.k(concatenated_features).unsqueeze(1)  # [B, 1, D]
        v = self.v(concatenated_features).unsqueeze(1)  # [B, 1, D]

        # 计算 attention 权重
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # [B, 1, 1]
        attn_weights = attn_weights / (concatenated_features.size(-1) ** 0.5)  # 缩放
        
        attn_weights = torch.softmax(attn_weights, dim=-1)  # Softmax 归一化
        # attn_weights = self.dropout(attn_weights)  # Dropout 防止过拟合

        # Step 3: 使用 attention 权重进行加权聚合
        attn_out = torch.bmm(attn_weights, v)  # [B, 1, D]

        # Step 4: 计算聚类输出和对比学习输出
        strong_features = attn_out.squeeze(1)[:, :strong_enhanced.size(-1)]  # [B, D] 对应强增强特征
        weak_features = attn_out.squeeze(1)[:, strong_enhanced.size(-1):]  # [B, D] 对应弱增强特征

        # # 处理聚类任务和对比学习任务
        # cluster_out = self.out_clustering(strong_features)  # 聚类任务的输出
        # contrastive_out = self.out_contrastive(weak_features)  # 对比学习任务的输出

        return strong_features, weak_features



class bestCNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.graph = Feat2Graph(nfeat)

        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc_pol = nn.Linear(2048,1024)
        self.bn = torchvision.ops.FrozenBatchNorm2d(num_features=1024)
        self.gn = nn.GroupNorm(num_groups=256, num_channels=2048)
    # def bottleneck(self, path1, path2, path3, adj, in_x):
    #     return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        x = F.relu(self.bn(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc_pol(x)
        return x

class netD_da(nn.Module):
    def __init__(self, feat_d):
        super(netD_da, self).__init__()
        self.fc1 = nn.Linear(feat_d,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x  #[256, 2]
    
