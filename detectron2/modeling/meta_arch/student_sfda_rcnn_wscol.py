# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import random
import os
from scipy.spatial.distance import cdist
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.structures import ImageList, Instances, pairwise_iou, Boxes
from detectron2.layers import cat
from torchvision.ops import nms  # BC-compat
import pickle
from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
import pdb
import cv2
import torch.nn.functional as F
from matplotlib import pyplot as plt
from .losses import GraphConLoss,con_loss,gradient_discrepancy_loss,simclr
# from .IID_losses import IID_loss
from .GCN import GCN,CNN
# from sklearn.cluster import KMeans
from torch.autograd import Variable
from.memorybank import MemoryBank
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from ..enhance.enhance_vgg16 import enhance_vgg16
import networkx as nx

__all__ = ["student_sfda_RCNN_wscol"]
@META_ARCH_REGISTRY.register()
class student_sfda_RCNN_wscol(nn.Module):
    """
    student_sfda_RCNN R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        cfg,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 10,
        memory_len: int = 10,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        if cfg.METHOD != 'MT':
            self.con_loss = con_loss
            self.CNN = CNN()
            self.fc_pol = nn.Linear(2048,512)
            self.relu = F.relu
            self.init = torch.ones(roi_heads.num_classes, device='cuda') #class_n 个 1
            self.memory_bank = MemoryBank(roi_heads.num_classes,memory_len)
            self.criterion = nn.L1Loss(reduction='none')
            self.prototype_pol = torch.zeros([roi_heads.num_classes,512],dtype=torch.float,device='cuda')
            self.num_class = roi_heads.num_classes

        if cfg.METHOD == 'wsco+IRG':
            self.GraphCN = GCN(nfeat=2048, nhid=512)
            self.Graph_conloss = GraphConLoss()
            self.MSEloss = nn.MSELoss()

        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        self.simclr = simclr()
        self.ce_loss = nn.CrossEntropyLoss()

        if cfg.METHOD == 'wsco+LODS':
            self.gw = gromovWasserstein(0.5) #GromovWasserstein (GW) 
            self.adain = enhance_vgg16(cfg)

        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "cfg": cfg,

            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "memory_len": cfg.MODEL.MEMORYLEN,

        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def image_vis(self, images):
        img = images[0].cpu().permute(1, 2, 0).numpy()
        cv2.imshow('img', img)
        cv2.waitKey(2500)
        pdb.set_trace()
    
    def KD_loss(self, student_logits, teacher_logits) :
        teacher_prob = F.softmax(teacher_logits, dim=1)
        student_log_prob = F.log_softmax(student_logits, dim=1)
        KD_loss = F.kl_div(student_log_prob, teacher_prob.detach(), reduction='batchmean')

        return KD_loss
    def sim(self,f, k):
    # Example similarity function: cosine similarity
        return F.cosine_similarity(f.unsqueeze(1), k.unsqueeze(0), dim=-1)

    
    def pro_const_loss(self,fea_s,fea_t,prototype):

        prob_s = F.cosine_similarity(fea_s.unsqueeze(1), prototype.unsqueeze(0), dim=-1)
        prob_t = F.cosine_similarity(fea_t.unsqueeze(1), prototype.unsqueeze(0), dim=-1)

        return self.KD_loss(prob_s,prob_t)

    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def feat_kd_high_loss(self, feat, teacher_feat):
        bs, c, h, w = feat.shape
        flatten_feat = feat.reshape(bs, c, -1)
        flatten_teacher_feat = teacher_feat.reshape(bs, c, -1)

        stu_aff = torch.bmm(flatten_feat.permute(0, 2, 1), flatten_feat)
        tea_aff = torch.bmm(flatten_teacher_feat.permute(0, 2, 1), flatten_feat).detach()

        #tea_aff = torch.bmm(flatten_teacher_feat.permute(0, 2, 1), flatten_teacher_feat)
        
        loss = self.criterion(stu_aff, tea_aff).mean()
        
        return loss
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], cfg=None, model_teacher=None, t_features=None, t_proposals=None, t_results=None,epoch=None,weight=None,emaweight=None,move=None,cluster_step=None,temp=None,estimate=None,mode="test"):

        if not self.training and mode == "test":
            return self.inference(batched_inputs,)

        if cfg.METHOD == 'wsco+LODS':
            images = self.preprocess_image(batched_inputs, 'esm')
            images.tensor = self.adain.add_style(images.tensor, 0)
        else:
            images = self.preprocess_image(batched_inputs, mode)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            # proposals, proposal_losses = self.proposal_generator(images, features,t_results)
            proposals, proposal_losses = self.proposal_generator(images, features,t_results)
            # proposals_weak, proposal_losses_weak = self.proposal_generator(images, features,t_results_low)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        results, detector_losses = self.roi_heads(images, features, proposals,t_results) #>0.5前景

        losses = {}


        grad_loss_scale = 1e4
        cluster_acc = 0
        pse_acc = 0

        # if estimate == True:
        #     return losses
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        # s_box_features_weak = self.roi_heads._shared_roi_transform([features_weak['res4']], [t_proposals[0].proposal_boxes]) #t_proposals[0], results[1]
        s_box_features = self.roi_heads._shared_roi_transform([features['res4']], [t_proposals[0].proposal_boxes]) #t_proposals[0], results[1]
        s_roih_logits = self.roi_heads.box_predictor(s_box_features.mean(dim=[2, 3]))
        # self.visualize_proposals_idx(cfg,batched_inputs,t_proposals,0,'rpn',MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),'s','1')
        t_box_features = model_teacher.roi_heads._shared_roi_transform([t_features['res4']], [t_proposals[0].proposal_boxes])
        t_roih_logits = model_teacher.roi_heads.box_predictor(t_box_features.mean(dim=[2, 3]))
        losses["st_const"] = self.KD_loss(s_roih_logits[0], t_roih_logits[0]) * 1.0
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if cfg.METHOD == 'wsco+IRG':
            s_graph_feat = self.GraphCN(s_box_features.mean(dim=[2, 3]))
            s_graph_logits = self.roi_heads.box_predictor(s_graph_feat)

            t_graph_feat = self.GraphCN(t_box_features.mean(dim=[2, 3]))
            t_graph_logits = model_teacher.roi_heads.box_predictor(t_graph_feat)

            # losses["st_const"] = self.KD_loss(s_roih_logits[0], t_roih_logits[0]) 
            losses["s_graph_const"] = self.KD_loss(s_graph_logits[0], s_roih_logits[0]) 
            losses["t_graph_const"] = self.KD_loss(t_graph_logits[0], t_roih_logits[0]) 
            losses["graph_conloss"] = self.Graph_conloss(t_box_features.mean(dim=[2, 3]), s_box_features.mean(dim=[2, 3]), self.GraphCN)

        if 'wsco' in cfg.METHOD and len(t_results[0]._fields['gt_classes'])!=0:

            # t_roih_logits = self.roi_heads.box_predictor(t_box_features.mean(dim=[2, 3]))
            base_output = nn.Softmax(dim=1)(t_roih_logits[0])
            fg_index,fg_labels,bg_index,bg_label,pse_labels,pseudo_matrix = model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,t_results)
            _, pse_labels = torch.max(base_output, 1)

            # idx_bg = torch.where(pseudo_matrix==-1)
            # pseudo_matrix[idx_bg] = len(torch.unique(pseudo_matrix))

            pse_id = torch.where(t_results[0]._fields['scores'])[0]
            pse_f_no_pol = self.roi_heads._shared_roi_transform([t_features['res4']], [t_results[0]._fields['gt_boxes']])[pse_id].detach()
            pse_l = t_results[0]._fields['gt_classes'][pse_id].detach()
            self.memory_bank.update(pse_f_no_pol,pse_l)
            me_f,me_l = self.memory_bank.read(10)
            m_f = torch.stack(me_f).cuda()
            m_l = torch.tensor(me_l).cuda()
            m_f_pol = m_f.mean(dim=[2, 3])
            m_logits = model_teacher.roi_heads.box_predictor(m_f_pol)[0].detach()
            memory_cnn_pol = self.fc_pol(F.relu(self.CNN(m_f))).detach()

            s_cnn_feat = self.CNN(s_box_features)
            t_cnn_feat = self.CNN(t_box_features).detach()
            s_cnn_pol = self.fc_pol(F.relu(s_cnn_feat))
            t_cnn_pol = self.fc_pol(F.relu(t_cnn_feat)).detach()
            all_features = []
            all_features_pol = []
            all_labels = []

            memory_fea = torch.cat((t_box_features.mean(dim=[2, 3]),m_f_pol),dim=0).detach().clone()
            memory_logits = torch.cat((t_roih_logits[0],m_logits),dim=0).detach().clone()

            prototype_pol = self.prototype_update_pol(memory_cnn_pol.detach(),m_l,emaweight)

            pro_index = torch.where(self.init!=1)[0]

            dd_c ,labelset_c,all_labels= self.pro_obtain_label(t_cnn_pol.detach().clone(),t_roih_logits[0].detach().clone(),self.num_class + 1,prototype_pol,cluster_step,pro_index,fg_index,fg_labels)

            del m_f, t_cnn_feat
            torch.cuda.empty_cache()

            if epoch==1:
                dd ,labelset_m,label_clus= self.obtain_label(memory_fea.detach().clone(),memory_logits.detach().clone(),self.num_class + 1,2,fg_index,fg_labels,base_bg_select=None)
                if torch.equal(labelset_c, labelset_m):
                    dd = (1 - move ) * dd[:len(s_box_features)] + ( move ) * dd_c
                    pred_label = dd.argmin(dim=1)
                    predict = labelset_m[pred_label]
                    all_labels = predict.int()

            all_labels = torch.tensor(all_labels[:len(t_box_features)]).cuda().to(dtype=torch.int64)
            all_labels[fg_index] = fg_labels
            scores = torch.sigmoid(t_proposals[0]._fields['objectness_logits'])
            index = torch.where(scores > 0.5)
            scores = scores[index]
            p = t_proposals[0]._fields['proposal_boxes'].tensor[index]
            iou_thresholds = np.arange(0.0, 0.6, 0.1)
            proposal_variance = self.calculate_proposal_variance(p, scores, iou_thresholds)


            if estimate ==True:
                images_weak = self.preprocess_image(batched_inputs, 'esm')
                features_weak = self.backbone(images_weak.tensor)
                # fea_neg_idx = torch.where(pseudo_matrix.sum(dim=0, keepdim=True)==0)[0]
                s_box_features_weak = self.roi_heads._shared_roi_transform([features_weak['res4']], [t_proposals[0].proposal_boxes]) #t_proposals[0], results[1]

                # fea_simclr_weak = self.roi_heads._shared_roi_transform([features_weak['res4']], [t_results[0]._fields['gt_boxes']]) #t_proposals[0], results[1]
                # fea_simclr_weak = torch.cat((fea_simclr_weak,s_box_features_weak[fea_neg_idx]))
                # fea_simclr_weak = self.CNN(fea_simclr_weak)
                # fea_simclr_weak = self.fc_pol(F.relu(fea_simclr_weak))
                # fea_simclr_strong = self.roi_heads._shared_roi_transform([features['res4']], [t_results[0]._fields['gt_boxes']]) #t_proposals[0], results[1]
                # fea_simclr_strong = torch.cat((fea_simclr_strong,s_box_features[fea_neg_idx]))
                # fea_simclr_strong = self.CNN(fea_simclr_strong)
                # fea_simclr_strong = self.fc_pol(F.relu(fea_simclr_strong))
                # fea_pos_idx = torch.arange(len(t_results[0]._fields['gt_boxes']))

                for k, v in self.roi_heads.named_parameters():
                    v.requires_grad = False 

                s_cnn_feat_weak = self.CNN(s_box_features_weak)
                s_cnn_pol_weak = self.fc_pol(F.relu(s_cnn_feat_weak))

                s_roih_logits_weak = self.roi_heads.box_predictor(s_cnn_feat_weak)
                s_roih_logits = self.roi_heads.box_predictor(s_cnn_feat)

                ce1 = self.ce_loss(s_roih_logits_weak[0][fg_index],fg_labels)
                ce2 = self.ce_loss(s_roih_logits[0][fg_index],fg_labels)

                if len(pse_l)!=0:
                    idx_bg = torch.where(pseudo_matrix==-1)
                    pseudo_matrix[idx_bg] = len(torch.unique(pseudo_matrix))
                    fea_simclr = torch.cat((s_cnn_pol_weak,s_cnn_pol))
                    labels_unsup = torch.cat((pseudo_matrix,pseudo_matrix))

                    losses['unsup'] = 0.01 * self.con_loss(fea_simclr,labels_unsup,0.07)
                    losses['loss_merge_grad'] = 0.1 * gradient_discrepancy_loss(self, grad_loss_scale * ce1, grad_loss_scale * ce2)

                for k, v in self.roi_heads.named_parameters():
                    v.requires_grad = True

                del s_box_features_weak, s_cnn_feat_weak, s_roih_logits_weak,s_cnn_feat,
                torch.cuda.empty_cache()

            reliability_martrix = self.classify_samples(t_cnn_pol.detach().clone(), all_labels,1.0,merge=cfg.merge)

            # model_teacher.roi_heads.proposal_matcher.thresholds[1] = 0.5
            # gt_fg_index,gt_fg_labels,gt_bg_index,gt_bg_labels,gt_labels= model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,gt_instances)
            # fg_index,fg_labels,bg_index,bg_label,pse_labels = model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,t_results)
            # model_teacher.roi_heads.proposal_matcher.thresholds[1] = 0.5
            # print(m_l.shape,memory_cnn_pol.shape)
            # s_cnn_pol = torch.cat((s_cnn_pol,memory_cnn_pol), dim=0)
            # all_labels = torch.cat([all_labels, m_l], dim=0)
            fg = torch.where(all_labels!=self.num_class)[0]

            if len(fg) > 2 :
                if proposal_variance > 20:
                    losses['cont'] = weight * self.con_loss(s_cnn_pol[fg],all_labels[fg],0.07,reliability_martrix[fg][:, fg],merge=cfg.merge,alpha=0.0)
                else:
                    losses['cont'] = weight * self.con_loss(s_cnn_pol,all_labels,0.07,reliability_martrix,merge=cfg.merge,alpha=0.0)

                # if proposal_variance > 20:
                #     losses['cont'] = 0.5 * self.con_loss(s_cnn_pol[fg],all_labels[fg],0.07)
                # else:
                #     losses['cont'] = 0.5 * self.con_loss(s_cnn_pol,all_labels,0.07)
            # losses['simclr'] = self.simclr(s_cnn_pol_weak,s_cnn_pol) * 0.1

            del s_cnn_pol, s_cnn_pol_weak, reliability_martrix
            torch.cuda.empty_cache()

            # model_teacher.roi_heads.proposal_matcher.thresholds[1] = 0.5
            gt_fg_index,gt_fg_labels,gt_bg_index,gt_bg_labels,gt_labels,_= model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,gt_instances)

            # # fg_index,fg_labels,bg_index,bg_label,pse_labels,_ = model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,t_results)
            # model_teacher.roi_heads.proposal_matcher.thresholds[1] = 0.5
            # if len(torch.unique(gt_fg_labels)) >=5:
            #     self.visualize_cnn_feats(t_cnn_pol[gt_fg_index].cpu(),gt_fg_labels.cpu(),save_path='cnn_feats_tsne.png')
            #     self.visualize_cnn_feats(t_box_features.mean(dim=[2, 3])[gt_fg_index].cpu(),gt_fg_labels.cpu(),save_path='ori_feats_tsne.png')
            #     print(1)

            if len(gt_fg_index) !=0 :
                cluster_acc = len(torch.where(gt_labels[gt_fg_index] == all_labels[gt_fg_index])[0]) / len(gt_fg_index)
                pse_acc = len(torch.where(gt_labels[gt_fg_index] == pse_labels[gt_fg_index])[0]) / len(gt_fg_index)
                # base_acc = len(torch.where(gt_labels[gt_fg_index] == base_l[gt_fg_index])[0]) / len(gt_fg_index)
                # shot_acc = len(torch.where(gt_labels[gt_fg_index] == all_labels_shot[gt_fg_index])[0]) / len(gt_fg_index)


        if cfg.METHOD == 'wsco+LPLD':
            s_box_features = self.roi_heads._shared_roi_transform([features['res4']], [t_proposals[0].proposal_boxes]) #s_box_features = 300,2048,7,7
            s_box_features_mean = s_box_features.mean(dim=[2, 3])
            s_box_features_norm = F.normalize(s_box_features_mean, dim=1)
            s_roih_logits = self.roi_heads.box_predictor(s_box_features_mean)

            t_box_features = model_teacher.roi_heads._shared_roi_transform([t_features['res4']], [t_proposals[0].proposal_boxes])
            t_box_features_mean = t_box_features.mean(dim=[2, 3])
            t_box_features_norm = F.normalize(t_box_features_mean, dim=1)
            t_roih_logits = model_teacher.roi_heads.box_predictor(t_box_features_mean)

            # Compute the cosine similarity between the student and teacher features
            c_similarity = F.cosine_similarity(s_box_features_norm.detach(), t_box_features_norm.detach(), dim=1)
            t_roih_classes = t_roih_logits[0].argmax(dim=1)

            # Compute the LPL loss
            t_proposal_boxes = cat([p.proposal_boxes.tensor for p in t_proposals], dim=0)
            t_boxes = model_teacher.roi_heads.box_predictor.box2box_transform.apply_deltas(t_roih_logits[1], t_proposal_boxes)
            t_box = torch.zeros(t_proposal_boxes.shape[0], 4).cuda()

            for index, cl in enumerate(t_roih_classes):
                if cl == self.roi_heads.num_classes:
                    t_box[index] = t_proposal_boxes[index]
                else:
                    t_box[index] =  t_boxes[index, 4*cl:4*cl+4]

            t_box_boxes = Boxes(t_box)
            t_result_boxes = Boxes(t_results[0].gt_boxes.tensor)
            iou_matrix = pairwise_iou(t_box_boxes, t_result_boxes)

            if iou_matrix.shape[1] == 0:
                return losses,cluster_acc,pse_acc

            t_indices = torch.nonzero(torch.max(iou_matrix, dim=1).values <= 0.4).flatten()
            if t_indices.nelement() == 0:
                return losses,cluster_acc,pse_acc

            t_softmax_w_bg = F.softmax(t_roih_logits[0][t_indices], dim=1)
            t_softmax_wo_bg = F.softmax(t_roih_logits[0][t_indices][:, :-1], dim=1)
            wo_bg_max = torch.max(t_softmax_wo_bg, dim=1)[0]
            w_bg_prob = t_softmax_w_bg[:, -1]
            t_indices_filtered = t_indices[(wo_bg_max >= 0.9) & (w_bg_prob <= 0.99)]

            if t_indices_filtered.nelement() == 0:
                return losses,cluster_acc,pse_acc

            losses["lpl_kl"] = self._kl_divergence(s_roih_logits[0][t_indices_filtered],
                                                t_roih_logits[0][t_indices_filtered][:, :-1],
                                                weight = 1-c_similarity[t_indices_filtered])

        if cfg.METHOD == 'wsco+LODS':
        # gw loss
            cls_score_s = s_roih_logits[0]
            cls_score_t = t_roih_logits[0]
            cls_prob_s = F.softmax(s_roih_logits[0],1)
            cls_prob_t = F.softmax(t_roih_logits[0],1)
            max_prob_t, labels_t = cls_prob_t.max(1)  #伪标签

            cls_prob_s_d = cls_prob_s
            cls_prob_t_d = cls_prob_t
            feat_s=torch.bmm(cls_prob_s_d.unsqueeze(2), s_box_features.mean(dim=[2, 3]).unsqueeze(1)).view(-1, cls_prob_s_d.size(1) * s_box_features.mean(dim=[2, 3]).size(1))
            feat_t=torch.bmm(cls_prob_t_d.unsqueeze(2), t_box_features.mean(dim=[2, 3]).unsqueeze(1)).view(-1, cls_prob_t_d.size(1) * t_box_features.mean(dim=[2, 3]).size(1))

            unit_matrix = torch.eye(self.num_class + 1).to(self.device)
            unit_matrix[-1][-1]=0 # ignore background

            bk_index = (max_prob_t<0.8).nonzero().squeeze(1)
            labels_t[bk_index]=0
            labels_t_onehot = unit_matrix[labels_t]
            mask = labels_t_onehot.mm(labels_t_onehot.t())
            # compute t
            cls_score_t_norm = cls_score_t/torch.sqrt((cls_score_t**2).sum(1,keepdim=True))
            t = cls_score_t_norm.mm(cls_score_t_norm.t()).detach()
            t = mask * t

            gw_loss = self.gw(feat_s,feat_t,t)

            shape = features['res4'].size()
            gw_global_loss = self.gw(torch.t(features['res4'].view(shape[0]*shape[1],shape[2]*shape[3])),torch.t(t_features['res4'].view(shape[0]*shape[1],shape[2]*shape[3])),0)
            gw_loss = 0.1 * gw_loss
            gw_global_loss = 0.1 * gw_global_loss
            losses["gw"] = gw_loss + gw_global_loss

        return losses,cluster_acc,pse_acc

    def _kl_divergence(self, student_logits, teacher_logits, weight=None):
        teacher_probs = F.softmax(teacher_logits, dim=1)
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = torch.cat([teacher_probs, 1e-10*torch.ones(teacher_probs.size(0), 1).cuda()], dim=1)
        KL_loss = teacher_probs * (teacher_probs.log() - student_probs.log())
        if weight is not None:
            KL_loss = (KL_loss.sum(1)*weight).sum()/student_probs.size(0)
        else:
            KL_loss = KL_loss.sum()/student_probs.size(0)

        return KL_loss/10
    
    def calculate_proposal_variance(self,proposals, objectness_scores, iou_thresholds):
        from torchvision.ops import nms  # BC-compat

        proposal_counts = []
        for iou_threshold in iou_thresholds:
            keep = nms(proposals, objectness_scores, iou_threshold)
            proposal_counts.append(len(keep))
        proposal_variance = np.var(proposal_counts)
        return proposal_variance

    def prototype_update_pol(self,fea,labels,ema):

        classes = torch.unique(labels)
        for i in range(classes.shape[0]):
            fea_mean = fea[labels==classes[i]].mean(dim=0)
            if self.init[classes[i]]:
                self.prototype_pol[classes[i]] = fea_mean
                self.init[classes[i]] = False
            else:
                self.prototype_pol[classes[i]] = (1 - ema) * self.prototype_pol[classes[i]] + \
                                                ema * fea_mean

        return self.prototype_pol
    
    def prototype_update(self,fea,labels,ema):

        classes = torch.unique(labels)
        for i in range(classes.shape[0]):
            fea_mean = fea[labels==classes[i]].mean(dim=0)
            if self.init[classes[i]]:
                self.prototype[classes[i]] = fea_mean
                self.init[classes[i]] = False
            else:
                self.prototype[classes[i]] = (1 - ema) * self.prototype[classes[i]] + \
                                                ema * fea_mean

        return self.prototype


    def reset_memory_bank(self): 
        self.memory_bank.reset()

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _, = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return student_sfda_RCNN_wscol._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]], mode = "test"):
        """
        Normalize, pad and batch the input images.
        """
        if mode == "train":
            images = [x["image_strong"].to(self.device) for x in batched_inputs]
            #self.image_vis(images)
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        elif mode == "test":
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        elif mode == "esm":
            images = [x["image_weak"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def visualize_cnn_feats(self, cnn_feat, labels, save_path='cnn_feats_tsne.png'):
        from sklearn.preprocessing import normalize
        """
        参数:
            cnn_feat: torch.Tensor 或 numpy.ndarray, 形状 [N, D]
            labels: list 或 numpy.ndarray, 形状 [N]
            save_path: 保存路径
        """
        # 转成 numpy 数组
        if isinstance(cnn_feat, torch.Tensor):
            feat_out = cnn_feat.detach().cpu().numpy()
        else:
            feat_out = cnn_feat

        # L2 归一化
        feat_out = normalize(feat_out, norm='l2', axis=1)

        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42)
        feat_2d = tsne.fit_transform(feat_out)

        # 绘图
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            feat_2d[:, 0],
            feat_2d[:, 1],
            c=labels,
            cmap='Set1',
            s=6,
            alpha=0.7
        )

        plt.title("t-SNE of CNN Features")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True)
        plt.legend(*scatter.legend_elements(), title="Classes", loc='upper right', fontsize='small')
        plt.tight_layout()

        # 保存图像
        plt.savefig(save_path, dpi=300)
        print(f"Saved t-SNE plot for CNN features to: {save_path}")


    def obtain_label(self, all_fea, aff, K, step, fg_index, fg_labels, base_bg_select):
        device = all_fea.device

        all_output = F.softmax(aff, dim=1)          # [N, K]
        _, predict = torch.max(all_output, 1)       # [N]

        # Append a column of ones and normalize
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1, device=device)), dim=1)  # [N, D+1]
        all_fea = all_fea / (all_fea.norm(p=2, dim=1, keepdim=True) + 1e-8)  # L2 normalize each row

        aff = all_output.clone().detach()  # [N, K]
        predict = predict.clone().detach()  # [N]
        fg_index = fg_index.clone().detach()
        fg_labels = fg_labels.clone().detach()

        for i in range(step):
            initc = aff.T @ all_fea
            # initc = torch.matmul(aff.T, all_fea)      # [K, D]
            initc = initc / (1e-8 + aff.sum(dim=0, keepdim=True).T)  # [K, 1]

            cls_count = torch.eye(K, device=device)[predict].sum(dim=0)  # [K]
            labelset = torch.nonzero(cls_count > 0).squeeze()  # [M] where M <= K

            # Compute cosine distance: 1 - cosine_similarity
            fea_norm = F.normalize(all_fea, p=2, dim=1)
            cent_norm = F.normalize(initc[labelset], p=2, dim=1)
            cosine_sim = torch.matmul(fea_norm, cent_norm.T)  # [N, M]
            dd = 1 - cosine_sim  # cosine distance

            pred_label = dd.argmin(dim=1)  # [N]
            predict = labelset[pred_label]  # [N]
            predict[fg_index] = fg_labels
            aff = torch.eye(K, device=device)[predict]  # [N, K]

        d = dd
        l = labelset
        pred_label = dd.argmin(dim=1)
        predict = labelset[pred_label]
        label = predict.int()

        return d, l, label
    

    def find_topN_samples(self,all_features, sample_feature, N):
        """
        计算一个样本的 TopN 相似样本
        """
        # 计算当前样本与所有样本的相似度（余弦相似度）
        similarity_matrix = torch.matmul(sample_feature.unsqueeze(0), all_features.T)
        # 获取 TopN 最相似的样本的索引
        topN_indices = torch.topk(similarity_matrix, N, dim=1).indices.squeeze(0)
        return topN_indices

    def classify_samples(self, features, labels, th_con=1.0, merge = None):
        N = features.shape[0]
        features = F.normalize(features, dim=1)
        if merge == 'weight':
            weight_matrix = torch.ones(N, N, device=features.device)
        if merge == 'merge':
            weight_matrix = torch.zeros(N, N, device=features.device)
            th_con = 1.0
        valid_idx = torch.where(labels != self.num_class)[0]

        for i in valid_idx:
            label_i = labels[i].item()
            same_class_mask = (labels == label_i)
            same_class_indices = same_class_mask.nonzero(as_tuple=False).squeeze(1)

            if same_class_indices.numel() <= 1:
                continue
            # 相似度并 Mask 掉自己
            sim = torch.matmul(features, features[i])
            sim_masked = sim.clone()
            sim_masked[i] = -1e9  # 排除自己

            # topK 同类数量 - 1
            topK = same_class_indices.numel() - 1
            topk_indices = torch.topk(sim_masked, k=topK).indices

            # trusted_pos_mask: 是 topk 且是同类
            topk_mask = torch.zeros(N, dtype=torch.bool, device=features.device)
            topk_mask[topk_indices] = True
            trusted_pos_mask = topk_mask & same_class_mask
            trusted_pos_mask[i] = False  # 去除自己
            # weight_matrix[i][trusted_pos_mask] = th_con

            # 非 trusted 的同类样本 -> 设置 th_con
            non_trusted_pos_mask = same_class_mask.clone()
            non_trusted_pos_mask[i] = False
            non_trusted_pos_mask &= ~trusted_pos_mask
            weight_matrix[i][non_trusted_pos_mask] = th_con

            # # 异类 & topk -> 设置 th_incon
            # diff_class_mask = (labels != label_i) & (labels != self.num_class)
            # incon_mask = diff_class_mask & topk_mask
            # weight_matrix[i][incon_mask] = th_incon

        return weight_matrix

    def pro_obtain_label(self, all_fea, aff, K, prototype, step, pro_index,fg_index,fg_labels):
        act = False
        if len(pro_index) == self.num_class:
            act = True

        # softmax and max prediction
        all_output = F.softmax(aff, dim=1)
        _, predict = torch.max(all_output, 1)

        # concatenate prototype and all_fea
        all_fea = torch.cat((prototype, all_fea), dim=0)
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1, device=all_fea.device)), dim=1)

        # normalize each row (features)
        all_fea = all_fea / (all_fea.norm(p=2, dim=1, keepdim=True))  # L2 normalization

        # convert to tensor and detach for further computation
        all_fea = all_fea.float()
        aff = all_output.float()

        # split prototype and all_fea
        prototype = all_fea[:K-1]
        all_fea = all_fea[K-1:]

        for i in range(step):
            # calculate initc using torch.matmul
            initc = aff.T @ all_fea
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

            # if first step and act is True, set initc[:K-1] to prototype
            if i == 0 and act:
                initc[:K-1] = prototype

            # get the labelset where class counts are greater than 0
            cls_count = torch.eye(K, device=all_fea.device)[predict].sum(dim=0)
            labelset = torch.nonzero(cls_count > 0).squeeze()
            # compute cosine distance

            if labelset.dim() ==0 :
                break
            dd = torch.cdist(all_fea, initc[labelset], p=2)  # use torch.cdist for cosine distance

            # update predictions based on the closest prototype
            pred_label = dd.argmin(dim=1)
            predict = labelset[pred_label]
            predict[fg_index] = fg_labels

            aff = torch.eye(K, device=all_fea.device)[predict]

        # final calculation of distances and labels
        d = dd
        l = labelset
        pred_label = dd.argmin(dim=1)
        predict = labelset[pred_label]
        label = predict.int()

        return d, l, label
    def visualize_proposals(self, cfg, batched_inputs, proposals, start,end, proposal_dir, metadata,aug):
            from detectron2.utils.visualizer import Visualizer

            for input, prop in zip(batched_inputs, proposals):
                if aug =='w':
                    img = input['image_weak']
                if aug =='s':  
                    img = input['image_strong']
                if aug =='style':  
                    img = input['image_weak']
                    img = self.adain.add_style(img, 0)


                img = convert_image_to_rgb(img.permute(1, 2, 0), None)
                # v_gt = Visualizer(img, metadata)
                # v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
                # anno_img = v_gt.get_image()
                v_pred = Visualizer(img, metadata)
                if proposal_dir == "rpn":
                    v_pred = v_pred.overlay_instances( boxes=prop.proposal_boxes[start:end].tensor.cpu().numpy())
                if proposal_dir == "pesudo":
                    v_pred = v_pred.overlay_instances( boxes=prop.gt_boxes[start:end].tensor.cpu().numpy())
                if proposal_dir == "roih":
                    v_pred = v_pred.draw_instance_predictions(prop)
                if proposal_dir == "gt":
                    v_pred = v_pred.overlay_instances( boxes = batched_inputs[0]['instances']._fields['gt_boxes'].tensor.cpu().numpy())
                vis_img = v_pred.get_image()

                save_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir) 
                save_img_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir, input['file_name'].split('/')[-1]) 
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_img_path, vis_img)
                
    def generate_knn_labels_with_graph(self,features, top_k=1):
        """
        通过图的连通分量生成伪标签
        :param features: Tensor [N, D] - 特征
        :param top_k: int - 每个样本连接的最近邻数
        :return: pseudo_labels [N] - 根据图的连通分量生成的类别标签
        """
        N = features.size(0)
        
        # 计算所有样本对之间的欧几里得距离
        dist = torch.cdist(features, features)  # [N, N]
        
        # 构建无向图
        graph = nx.Graph()
        graph.add_nodes_from(range(N))

        # 为每个样本添加最近邻连接
        for i in range(N):
            dists = dist[i].clone()
            dists[i] = float('inf')  # 去掉自己
            knn = torch.topk(dists, k=top_k, largest=False).indices.tolist()
            for j in knn:
                graph.add_edge(i, j)

        # 提取图的连通分量
        components = list(nx.connected_components(graph))
        
        # 为每个连通分量分配一个伪标签
        pseudo_labels = torch.full((N,), -1, dtype=torch.long)
        for idx, comp in enumerate(components):
            for node in comp:
                pseudo_labels[node] = idx  # 将连通分量的节点标记为同一类别
        
        return pseudo_labels
    
    def visualize_proposals_idx(self, cfg, batched_inputs, proposals, idx, proposal_dir, metadata,aug,color):
            from detectron2.utils.visualizer import Visualizer

            for input, prop in zip(batched_inputs, proposals):
                if aug =='w':
                    img = input['image_weak']
                if aug =='s':  
                    img = input['image_strong']
                if aug =='style':  
                    img = input['image_weak']
                    img = self.adain.add_style(img, 0)
                img = convert_image_to_rgb(img.permute(1, 2, 0), None)
                # v_gt = Visualizer(img, metadata)
                # v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
                # anno_img = v_gt.get_image()
                v_pred = Visualizer(img, metadata)
                if proposal_dir == "rpn":
                    v_pred = v_pred.overlay_instances( boxes=prop.proposal_boxes[idx].tensor.cpu().numpy(),color=color)
                if proposal_dir == "pesudo":
                    v_pred = v_pred.overlay_instances( boxes=prop.gt_boxes[idx].tensor.cpu().numpy())
                if proposal_dir == "roih":
                    v_pred = v_pred.draw_instance_predictions(prop)
                if proposal_dir == "gt":
                    v_pred = v_pred.overlay_instances( boxes = batched_inputs[0]['instances']._fields['gt_boxes'].tensor.cpu().numpy(),color=color)
                vis_img = v_pred.get_image()

                save_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir) 
                save_img_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir, input['file_name'].split('/')[-1]) 
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_img_path, vis_img)

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    

def vis_matrix(fea):
        
        feats = fea[:25].detach().clone().cpu().numpy()
        corr = np.corrcoef(feats)
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()


class Adaparams(nn.Module):
    def __init__(self, depth=10):
        super(Adaparams, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.depth = depth
        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()
        for i in range(depth):
            self.weight.append(nn.Parameter(torch.ones(2048)))
            self.bias.append(nn.Parameter(torch.zeros(2048)))

    def forward(self, x):
        for i in range(self.depth-1):
            x = self.relu(self.weight[i] * x + self.bias[i])
        x = self.weight[i+1] * x + self.bias[i+1]
        return x




import torch.nn as nn
import torch 
from torch.autograd import Variable
class gromovWasserstein(nn.Module):

    def __init__(self,beta=0.5,affinity_type='cosine',l_type='KL'):
        super(gromovWasserstein, self).__init__()
        self.affinity_type=affinity_type
        self.l_type=l_type
        self.beta = beta
        self.rate = 0.99
        self.iter_num = 50
        print("gw add rate is :"+str(self.beta))
    
    def forward(self,feat_stu,feat_tea,t):
        affinity_stu = self.affinity_matrix(feat_stu)
        affinity_tea = self.affinity_matrix(feat_tea)
        T = torch.eye(feat_stu.size(0)).cuda()

        if type(t)!=int:
            T = self.beta*t + T
        T = T/T.sum()
        cost = self.L(affinity_stu,affinity_tea,T)
        loss = (cost * T).sum()
        return loss
    
    def affinity_matrix_cross(self,feat1,feat2):
        if self.affinity_type=='cosine':
            energy1 = torch.sqrt(torch.sum(feat1 ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            energy2 = torch.sqrt(torch.sum(feat2 ** 2, dim=1, keepdim=True))
            cos_sim = torch.matmul(feat1, torch.t(feat2)) / (torch.matmul(energy1, torch.t(energy2)))
            affinity = cos_sim
        else:
            pass
        return affinity

    def affinity_matrix(self,feat):
        if self.affinity_type=='cosine':
            energy = torch.sqrt(torch.sum(feat ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            cos_sim = torch.matmul(feat, torch.t(feat)) / (torch.matmul(energy, torch.t(energy)) )
            affinity = cos_sim
        else:
            feat = torch.matmul(feat, torch.t(feat))  # (batch_size, batch_size)
            feat_diag = torch.diag(feat).view(-1, 1).repeat(1, feat.size(0))  # (batch_size, batch_size)
            affinity = 1-torch.exp(-(feat_diag + torch.t(feat_diag) - 2 * feat)/feat.size(1))
        return affinity
    
    def L(self,affinity_stu,affinity_tea,T):
        stu_1 = Variable(torch.ones(affinity_stu.size(0),1).cuda())
        tea_1 = Variable(torch.ones(affinity_tea.size(0),1).cuda())
        p=T.mm(tea_1)
        q=T.t().mm(stu_1)
        if self.l_type == 'L2':
            # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
            # cost_st = f1(affinity_stu)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(affinity_tea)^T
            # cost = cost_st - h1(affinity_stu)*T*h2(affinity_tea)^T
            f1_st = (affinity_stu ** 2).mm(p).mm(tea_1.t())  
            f2_st = stu_1.mm(q.t()).mm((affinity_tea ** 2).t())
            cost_st = f1_st + f2_st
            cost = cost_st - 2 * affinity_stu.mm(T).mm(affinity_tea.t())
        elif self.l_type=='KL':
            # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
            # cost_st = f1(affinity_stu)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(affinity_tea)^T
            # cost = cost_st - h1(affinity_stu)*T*h2(affinity_tea)^T
            f1_st = torch.matmul(affinity_stu * torch.log(affinity_stu+ 1e-7) - affinity_stu, p).mm(tea_1.t())
            f2_st = stu_1.mm(torch.matmul(torch.t(q), torch.t(affinity_tea)))
            cost_st = f1_st + f2_st
            cost = cost_st - torch.matmul(torch.matmul(affinity_stu, T), torch.t(torch.log(affinity_tea+1e-7)))
        return cost