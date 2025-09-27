"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable,Function
from torch.autograd import grad

class GraphConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07*2):
        super(GraphConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        dim_in = 2048
        feat_dim = 2048
        self.head_1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        self.head_2 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
    def forward(self, t_feat, s_feat, graph_cn, labels=None, mask=None):    

        qx = graph_cn.graph.wq(s_feat)
        kx = graph_cn.graph.wk(s_feat)
        sim_mat = qx.matmul(kx.transpose(-1, -2))
        dot_mat = sim_mat.detach().clone()

        thresh = 0.5
        dot_mat -= dot_mat.min(1, keepdim=True)[0]
        dot_mat /= dot_mat.max(1, keepdim=True)[0]  # normal
        mask = ((dot_mat>thresh)*1).detach().clone()
        mask.fill_diagonal_(1)

        anchor_feat = self.head_1(s_feat)
        contrast_feat = self.head_2(s_feat)

        anchor_feat = F.normalize(anchor_feat, dim=1) #pos
        contrast_feat = F.normalize(contrast_feat, dim=1) #neg

        ss_anchor_dot_contrast = torch.div(torch.matmul(anchor_feat, contrast_feat.T), self.temperature)  ##### torch.Size([6, 6])       similarity
        logits_max, _ = torch.max(ss_anchor_dot_contrast, dim=1, keepdim=True)  ##### torch.Size([6, 1]) - contains max value along dim=1
        ss_graph_logits = ss_anchor_dot_contrast - logits_max.detach()

        ss_graph_all_logits = torch.exp(ss_graph_logits) #
        ss_log_prob = ss_graph_logits - torch.log(ss_graph_all_logits.sum(1, keepdim=True)) #
        ss_mean_log_prob_pos = (mask * ss_log_prob).sum(1) / mask.sum(1)
    
        # loss
        ss_loss = - (self.temperature / self.base_temperature) * ss_mean_log_prob_pos
        ss_loss = ss_loss.mean()

        return ss_loss

def con_loss_weight(s_feat, labels, temp, pos_weight_matrix=None,bg_idx=None):

    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
    mask = label_matrix.float()
    
    if bg_idx is not None:
        mask[labels == bg_idx] = 0

    mask.fill_diagonal_(1)  # 将对角线设为1，表示每个样本是自己的正样本

    anchor_feat = F.normalize(s_feat, dim=1)  # 归一化特征
    ss_anchor_dot_contrast = torch.div(torch.matmul(anchor_feat, anchor_feat.T), temp)  # 计算相似度

    if pos_weight_matrix is not None:
        ss_anchor_dot_contrast = ss_anchor_dot_contrast * pos_weight_matrix

    logits_max, _ = torch.max(ss_anchor_dot_contrast, dim=1, keepdim=True)
    ss_graph_logits = ss_anchor_dot_contrast - logits_max.detach() 

    ss_graph_all_logits = torch.exp(ss_graph_logits) 
    ss_log_prob = ss_graph_logits - torch.log(ss_graph_all_logits.sum(1, keepdim=True))  # 对数概率

    eps = 1e-9


    ss_mean_log_prob_pos = (mask * ss_log_prob).sum(1) / mask.sum(1)
    ss_loss = - (temp / (temp * 2)) * ss_mean_log_prob_pos  

    index = torch.where(ss_loss != 0)[0]
    ss_loss = ss_loss[index].mean()  

    return ss_loss


def con_loss(s_feat, labels, temp, pos_weight_matrix=None,merge=None,alpha=0.5):

    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
    mask = label_matrix.float()

    mask.fill_diagonal_(1)  # 将对角线设为1，表示每个样本是自己的正样本

    anchor_feat = F.normalize(s_feat, dim=1)  # 归一化特征
    ss_anchor_dot_contrast = torch.div(torch.matmul(anchor_feat, anchor_feat.T), temp)  # 计算相似度

    if pos_weight_matrix is not None and merge=='weight':
        ss_anchor_dot_contrast = ss_anchor_dot_contrast * pos_weight_matrix

    logits_max, _ = torch.max(ss_anchor_dot_contrast, dim=1, keepdim=True)
    ss_graph_logits = ss_anchor_dot_contrast - logits_max.detach() 

    ss_graph_all_logits = torch.exp(ss_graph_logits) 
    ss_log_prob = ss_graph_logits - torch.log(ss_graph_all_logits.sum(1, keepdim=True))  # 对数概率

    eps = 1e-5

    if merge == 'merge' and pos_weight_matrix is not None and pos_weight_matrix.sum()!=0:

        mask_hp = pos_weight_matrix
        mask_nhp = mask * (1 - pos_weight_matrix)

        pos_hp = (mask_hp * ss_log_prob).sum(1) / (mask_hp.sum(1) + eps)
        posn_hp = (mask_nhp * ss_log_prob).sum(1) / (mask_nhp.sum(1)+ eps)

        loss_hp = - (temp / (temp * 2)) * pos_hp
        loss_nhp = - (temp / (temp * 2)) * posn_hp  

        index_hp = torch.where(loss_hp != 0)[0]
        loss_hp = loss_hp[index_hp].mean()  

        index_nhp = torch.where(loss_nhp != 0)[0]
        loss_nhp = loss_nhp[index_nhp].mean()  

        loss = alpha * loss_hp + (1 - alpha) * loss_nhp

        return loss
    else:
        ss_mean_log_prob_pos = (mask * ss_log_prob).sum(1) / mask.sum(1)
        ss_loss = - (temp / (temp * 2)) * ss_mean_log_prob_pos  
        index = torch.where(ss_loss != 0)[0]
        ss_loss = ss_loss[index].mean()  

    return ss_loss

# def merge_contrastive(loss_hp,loss_nhp,alpha = 0.5):
#     index_hp = torch.where(loss_hp != 0)[0]
#     loss_hp = loss_hp[index_hp].mean()  

#     index_nhp = torch.where(loss_nhp != 0)[0]
#     loss_nhp = loss_nhp[index_nhp].mean()  

#     loss = alpha * loss_hp + (1 - alpha) * loss_nhp

#     return loss

def con_loss_c(s_feat,labels,mask=None):    

    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
    mask = label_matrix.float()
    mask.fill_diagonal_(1)
    # anchor_feat = relu(pol(net(s_feat)))

    anchor_feat = F.normalize(s_feat, dim=1) #pos
    # contrast_feat = F.normalize(contrast_feat, dim=1) #neg

    ss_anchor_dot_contrast = torch.div(torch.matmul(anchor_feat, anchor_feat.T), 0.07)  ##### torch.Size([6, 6])       similarity
    logits_max, _ = torch.max(ss_anchor_dot_contrast, dim=1, keepdim=True)  ##### torch.Size([6, 1]) - contains max value along dim=1
    ss_graph_logits = ss_anchor_dot_contrast - logits_max.detach()

    ss_graph_all_logits = torch.exp(ss_graph_logits) #
    ss_log_prob = ss_graph_logits - torch.log(ss_graph_all_logits.sum(1, keepdim=True)) #
    ss_mean_log_prob_pos = (mask * ss_log_prob).sum(1) / mask.sum(1)

    # loss
    ss_loss = - (0.07 / (0.07*2)) * ss_mean_log_prob_pos
    ss_loss = ss_loss.mean()

    return ss_loss

def gradient_discrepancy_loss(model, lossa, lossb):
    grad_losses = []
    for n, p in model.CNN.named_parameters():
        if p.requires_grad==False:
            continue
        grad_a = grad([lossa],
            [p],
            create_graph=True,
            only_inputs=True,allow_unused=True)[0]

        grad_b = grad([lossb],
            [p],
            create_graph=True,
            only_inputs=True,allow_unused=True)[0]

        if grad_a is None or grad_b is None:
            continue

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(grad_a.detach(), grad_b, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(grad_a.detach(), grad_b, dim=0)
        grad_losses.append(_cossim)
    
    grad_losses = torch.stack(grad_losses)

    return (1.0 - grad_losses).mean()


class simclr(nn.Module):
    def __init__(self, temperature=0.07):
        super(simclr, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j, fg_idx = None,match_matrix=None):
        # 归一化特征
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
        z = nn.functional.normalize(z, dim=-1)

        similarity_matrix = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]

        # 计算相似度矩阵
        # similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature

        labels = torch.arange(len(z)).cuda()
        losses = nn.CrossEntropyLoss(reduction='none')(similarity_matrix, labels)

        # 只对前景样本的损失求平均
        if fg_idx is not None:
            N = z_i.size(0)
            fg_idx = torch.cat([fg_idx, fg_idx + N])
            fg_losses = losses[fg_idx]
            loss = fg_losses.mean()
        else:
            loss = losses.mean()
        assert torch.isfinite(loss).all()
        return loss