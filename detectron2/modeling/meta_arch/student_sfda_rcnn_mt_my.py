import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from scipy.spatial.distance import cdist
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
import pickle
from collections import deque
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
import pdb
import cv2
import torch.nn.functional as F
from matplotlib import pyplot as plt
from .losses import GraphConLoss,con_loss
from .IID_losses import IID_loss
from .GCN import GCN,CNN
# from sklearn.cluster import KMeans
from torch.autograd import Variable
from.memorybank import MemoryBank
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from typing import List
import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms ,box_iou # BC-compat

from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.structures import Boxes, Instances


__all__ = ["student_sfda_RCNN_mt_my"]
@META_ARCH_REGISTRY.register()
class student_sfda_RCNN_mt_my(nn.Module):
    """
    student_sfda_RCNN R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 10,
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

        # self.GraphCN = GCN(nfeat=2048, nhid=512)
        # self.Graph_conloss = GraphConLoss()
        # self.MSEloss = nn.MSELoss()
        # self.adaparams = Adaparams()
        self.con_loss = con_loss
        self.CNN = CNN()
        self.fc_pol = nn.Linear(2048,512)
        self.relu = F.relu
        # self.rational_bank = torch.zeros(9, 9, 2048, device='cpu')
        self.init = torch.ones(roi_heads.num_classes, device='cuda') #class_n 个 1
        # self.init_pol = torch.ones(8, device='cpu') #class_n 个 1
        self.memory_bank = MemoryBank(roi_heads.num_classes,10)
        # self.classifier = nn.Linear(512,9)
        # self.criterion = nn.L1Loss(reduction='none')
        # self.prototype = torch.zeros([roi_heads.num_classes,2048],dtype=torch.float,device='cuda')
        self.prototype_pol = torch.zeros([roi_heads.num_classes,512],dtype=torch.float,device='cuda')
        # self.variance_window = deque(maxlen=200)
        # self.cls_layer = nn.Linear(512,9)
        self.num_class = roi_heads.num_classes

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
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
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], cfg=None, model_teacher=None, t_features=None, t_proposals=None, t_results=None,t_results_high=None,epoch=None,weight=None,emaweight=None,move=None,cluster_step=None,temp=None,warm=None,mode="test"):

        if not self.training and mode == "test":
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs, mode)
        # images_weak = self.preprocess_image_weak(batched_inputs, mode)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # images.tensor = masking(images.tensor) # add masking

        features = self.backbone(images.tensor)

        # features_weak = self.backbone(images.tensor)

        # if self.proposal_generator is not None:
        #     proposals, proposal_losses = self.proposal_generator(images, features,t_results)
        # else:
        #     assert "proposals" in batched_inputs[0]
        #     proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        #     proposal_losses = {}

        # results, detector_losses = self.roi_heads(images, features, proposals,t_results) #>0.5前景

        # if self.vis_period > 0:
        #     storage = get_event_storage()
        #     if storage.iter % self.vis_period == 0:
        #         self.visualize_training(batched_inputs, proposals)

        losses = {}
        # losses.update(detector_losses)
        # losses.update(proposal_losses)

        # scores = torch.sigmoid(t_proposals[0]._fields['objectness_logits'])
        # low_index = torch.where(scores <= 0.05)[0]
        # s = self.score_proposals(t_proposals,keep)

        # if len(gt_instances[0]._fields['gt_classes']) <=5 : 
        #     print(1)
        #     visualize_proposals(cfg,batched_inputs,gt_instances,torch.arange(300),'gt',MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),'w')


        s_box_features = self.roi_heads._shared_roi_transform([features['res4']], [t_proposals[0].proposal_boxes]) #t_proposals[0], results[1]

        s_roih_logits = self.roi_heads.box_predictor(s_box_features.mean(dim=[2, 3]))
        # visualize_proposals(cfg,batched_inputs,t_proposals,0,'rpn',MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),'s')

        t_box_features = model_teacher.roi_heads._shared_roi_transform([t_features['res4']], [t_proposals[0].proposal_boxes])
        t_roih_logits = model_teacher.roi_heads.box_predictor(t_box_features.mean(dim=[2, 3]))

        # gt_box_features = model_teacher.roi_heads._shared_roi_transform([features['res4']], [gt_instances[0]._fields['gt_boxes']]) #t_proposals[0], results[1]

        # self.save_step_data(gt_box_features.mean(dim=[2, 3]),gt_instances[0]._fields['gt_classes'],cfg.OUTPUT_DIR)

        t_s_roih_logits = self.roi_heads.box_predictor(t_box_features.mean(dim=[2, 3]))

        fg_index,fg_labels,bg_index,bg_label,pse_labels = model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,t_results)
        fg_index_high,fg_labels_high,bg_index,bg_label,pse_labels = model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,t_results_high)


        # gt_fg_index,gt_fg_labels,gt_bg_index,gt_bg_labels,gt_labels= model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,gt_instances)
        num_classes = self.roi_heads.num_classes + 1

        #memory
        if warm==True:
            results = t_results_high
        else:
            results = t_results

        pse_id = torch.where(results[0]._fields['scores'])[0]
        pse_f_no_pol =  self.roi_heads._shared_roi_transform([t_features['res4']], [results[0]._fields['gt_boxes']])[pse_id].detach()
        # pse_fea =  pse_f_no_pol.mean(dim=[2, 3])

        pse_l = results[0]._fields['gt_classes'][pse_id].detach()
        # pse_logits = model_teacher.roi_heads.box_predictor(pse_fea)[0].detach()
        # pse_output = nn.Softmax(dim=1)(pse_logits)
        # _, pse_l = torch.max(pse_output, 1)
        self.memory_bank.update(pse_f_no_pol,pse_l)

        me_f,me_l = self.memory_bank.read(10)

        # me_f,me_l = self.memory_bank.read(10)
        m_f = torch.stack(me_f).cuda()
        m_l = torch.tensor(me_l).cuda()
        m_f_pol = m_f.mean(dim=[2, 3])

        m_logits = model_teacher.roi_heads.box_predictor(m_f_pol)[0].detach()

        # losses["cma"] = self.feat_kd_high_loss(features['res4'],t_features['res4']) * 0.001

        s_cnn_feat = self.CNN(s_box_features)
        t_cnn_feat = self.CNN(t_box_features).detach()

        s_cnn_pol = self.fc_pol(F.relu(s_cnn_feat))
        t_cnn_pol = self.fc_pol(F.relu(t_cnn_feat)).detach()
        memory_cnn_pol = self.fc_pol(F.relu(self.CNN(m_f))).detach()

        # prototype = self.prototype_update(m_f_pol,m_l,emaweight)
        # if len(pse_l)!=0:
        #     pseudo_pol = self.fc_pol(F.relu(self.CNN(pse_f_no_pol))).detach()
        #     prototype_pol = self.prototype_update_pol(pseudo_pol,pse_l,emaweight)
        # else:
        #     prototype_pol = self.prototype_pol
        prototype_pol = self.prototype_update_pol(memory_cnn_pol,m_l,emaweight)

        # s_cnn_feat = self.CNN(s_box_features)
        # t_cnn_feat = self.CNN(t_box_features).detach()
        # res5 = self.roi_heads.res5(t_features['res4'])

        # if warm == False:
        #     feat = self.CNN(res5 ,images_weak).detach().cpu()
        # visualize_map(res5, images_weak.tensor)

        # losses["cma"] = self.feat_kd_high_loss(features['res4'],t_features['res4']) * 0.001
        # losses["st_const"] = self.KD_loss(s_roih_logits[0], t_roih_logits[0])


        # if len(pse_l)!=0:
        #     pseudo_pol = self.fc_pol(F.relu(self.CNN(pse_f_no_pol))).detach()
        #     prototype_pol = self.prototype_update_pol(pseudo_pol,pse_l,emaweight)
        # else:
        #     prototype_pol = self.prototype_pol
        memory_fea = torch.cat((t_box_features.mean(dim=[2, 3]),m_f_pol),dim=0)
        # memory_fea_pol = self.fc_pol(F.relu(self.CNN(memory_fea)))


        memory_logits = torch.cat((t_roih_logits[0],m_logits),dim=0)
        # memory_logits_s = torch.cat((s_roih_logits[0],m_logits),dim=0)
        # clus_fea = memory_fea_pol
        # clus_logits = memory_logits

        # base_output = nn.Softmax(dim=1)(t_s_roih_logits[0])
        # _, base_l = torch.max(base_output, 1)

        # base_bg_index = torch.where(base_l==(num_classes-1))[0]
        # base_bg_logtis_top2,_ = torch.topk(base_output[base_bg_index],k=2)

        scores = torch.sigmoid(t_proposals[0]._fields['objectness_logits'])
        index = torch.where(scores > 0.5)
        scores = scores[index]
        p = t_proposals[0]._fields['proposal_boxes'].tensor[index]
        # # all_indices = torch.tensor([]).cuda()
        iou_thresholds = np.arange(0.0, 0.6, 0.1)

        # for iou_threshold in iou_thresholds:
        #     keep_indices = nms(p, scores, iou_threshold)
        #     all_indices = torch.cat((all_indices, keep_indices))
        # all_indices = torch.unique(all_indices).long()

        # containing_proposals = list(set(all_indices.tolist()))
        # if warm == False:
        #     visualize_proposals(cfg,batched_inputs,t_proposals,keep_indices,'rpn',MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),'s')

        proposal_variance = self.calculate_proposal_variance(p, scores, iou_thresholds)
        # smoothed_variance = self.moving_average(self.variance_window, proposal_variance, 0.1)

        # mean_variance, std_variance = self.update_sliding_window(self.variance_window, proposal_variance ,200)

        # adaptive_threshold = mean_variance + 0.2 * std_variance
        # print(adaptive_threshold)
        # pro_index = torch.where(self.init!=1)[0]

        dd_c ,labelset_c,all_labels,softmax_output = self.pro_obtain_label(t_cnn_pol,t_roih_logits[0],self.num_class + 1,prototype_pol,cluster_step,fg_index_high,fg_labels_high)
        # dd_c ,labelset_c,all_labels= self.obtain_label(t_box_features.mean(dim=[2, 3]),t_roih_logits[0],9,2)
        # all_labels_shot = torch.tensor(all_labels_shot[:len(t_box_features)]).cuda()

        # dd ,labelset_m,all_labels= self.obtain_label(memory_fea,memory_logits,self.num_class + 1,2,fg_index_high,fg_labels_high)
        cluster_acc,pse_acc = 0,0
        gt_fg_index,gt_fg_labels,gt_bg_index,gt_bg_labels,gt_labels= model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,gt_instances)

        # if len(pro_index) == 8:
        #     cluster_results = fast_rcnn_inference_single_image(t_proposals[0]._fields['proposal_boxes'].tensor,softmax_output[:300],images.image_sizes[0],0.05,0.5,100)
        #     pred_instances = model_teacher.roi_heads.forward_with_given_boxes(features, cluster_results) ##### send pred_proposal to mask head for instance segmentation
        #     cluster_pseudo_results, num_roih_proposal = process_pseudo_label(pred_instances, 0.3, "roih", "thresholding")

        if warm == True:
            dd ,labelset_m,label_clus= self.obtain_label(memory_fea,memory_logits,self.num_class + 1,2,fg_index_high,fg_labels_high)
            if np.array_equal(labelset_c, labelset_m):
                dd = (1 - move ) * dd[:len(s_box_features)] + ( move ) * dd_c

            pred_label = dd.argmin(axis=1)
            predict = labelset_m[pred_label]
            all_labels = predict.astype('int')

        # base_bg_index = torch.where(base_l==8)[0]
        # base_bg_logtis_top2,_ = torch.topk(base_output[base_bg_index],k=2)

        # base_bg_select = base_bg_index[torch.where(base_bg_logtis_top2[:,0] > 0.95)]
        all_labels = torch.tensor(all_labels[:300]).cuda()

        # all_labels = torch.tensor(all_labels[:len(t_box_features)]).cuda()
        # all_labels[low_index] = num_classes - 1
        all_labels[fg_index_high] = fg_labels_high
        # all_labels[300:] = m_l

        # cluster_acc ,base_acc,shot_acc=0,0,0
        # cluster_acc_all ,base_acc_all=0,0,
        # pse_acc,pse_acc_all = 0,0
        # acc_new =0

        model_teacher.roi_heads.proposal_matcher.thresholds[1] = 0.5
        fg_index,fg_labels,bg_index,bg_labels,pse_labels = model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,t_results)
        model_teacher.roi_heads.proposal_matcher.thresholds[1] = 0.5

        # all_labels[base_bg_select] = self.num_class
        all_labels[fg_index_high] = fg_labels_high

        cluster_acc,pse_acc,count = 0,0,0
        # fg_positions = torch.nonzero(fg_index.unsqueeze(0) == pse_idx.unsqueeze(1), as_tuple=True)[1]

        if len(gt_labels) !=0 and len(t_results[0]._fields['idx'])!=0:

            pse_idx_high = t_results_high[0]._fields['idx']

            pse_idx = t_results[0]._fields['idx']
            gt_idx = gt_labels[pse_idx]
            clustering_predict = all_labels[pse_idx]

            bg_idx = torch.where(clustering_predict == self.num_class)[0]

            t_predict = t_results[0]._fields['gt_classes']
            t_predict_high = t_results_high[0]._fields['gt_classes']

            # mask = ~torch.isin(pse_idx, fg_index)
            # not_in_fg_index = pse_idx[mask]
            # fg_index = torch.cat((fg_index, not_in_fg_index))
            # fg_positions = torch.nonzero(fg_index.unsqueeze(0) == pse_idx.unsqueeze(1), as_tuple=True)[1]

            # if len(fg_positions)!= len(gt_idx):
            #     print(1)
            # dd_c ,labelset_c,all_labels = self.pro_obtain_label(t_cnn_pol[fg_index],t_roih_logits[0][fg_index],self.num_class + 1,prototype_pol,cluster_step,fg_index_high,fg_labels_high)
            # all_labels = torch.tensor(all_labels).cuda()

            # clustering_predict = all_labels[fg_positions]

            diffirent_idx = torch.where((clustering_predict == t_predict) & (clustering_predict!=8))[0]
            consistance_idx = torch.arange(len(pse_idx_high)).cuda()
            train_idx = torch.where((clustering_predict == t_predict) & (clustering_predict!=8))[0]

            all_idx = torch.cat((consistance_idx, diffirent_idx),dim=0)

            if len(diffirent_idx)!=0:
                cluster_acc = len(torch.where(clustering_predict == gt_idx)[0]) / len(gt_idx)
                pse_acc = len(torch.where(t_predict == gt_idx)[0]) / len(gt_idx)
                count = len(diffirent_idx) - len(pse_idx_high)
            else:
                cluster_acc = len(torch.where(clustering_predict == gt_idx)[0]) / len(gt_idx)
                pse_acc = len(torch.where(t_predict == gt_idx)[0]) / len(gt_idx)

            # t_results[0]._fields['gt_classes'] = clustering_predict

            # if bg_idx is not None:
            #     t_results[0] = t_results[0][[i for i in range(len(t_results[0])) if i in train_idx]]

        # if warm == False:

        if self.proposal_generator is not None:
            # proposals, proposal_losses = self.proposal_generator(images, features,t_results)
            proposals, proposal_losses = self.proposal_generator(images, features,gt_instances)
            # proposals_weak, proposal_losses_weak = self.proposal_generator(images, features,t_results_low)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        results, detector_losses = self.roi_heads(images, features, proposals,gt_instances) #>0.5前景
        losses.update(detector_losses)
        losses.update(proposal_losses) 

        # else:
        #     if self.proposal_generator is not None:
        #         # proposals, proposal_losses = self.proposal_generator(images, features,t_results)
        #         proposals, proposal_losses = self.proposal_generator(images, features,t_results_high)
        #         # proposals_weak, proposal_losses_weak = self.proposal_generator(images, features,t_results_high_low)
        #     else:
        #         assert "proposals" in batched_inputs[0]
        #         proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        #         proposal_losses = {}

        #     results, detector_losses = self.roi_heads(images, features, proposals,t_results_high) #>0.5前景
        #     losses.update(detector_losses)
        #     losses.update(proposal_losses)

        # if len(gt_fg_index) !=0 :
        #     cluster_acc = len(torch.where(gt_labels[gt_fg_index] == all_labels[gt_fg_index])[0]) / len(gt_fg_index)
        #     pse_acc = len(torch.where(gt_labels[gt_fg_index] == pse_labels[gt_fg_index])[0]) / len(gt_fg_index)
        #     base_acc = len(torch.where(gt_labels[gt_fg_index] == base_l[gt_fg_index])[0]) / len(gt_fg_index)
        #     # shot_acc = len(torch.where(gt_labels[gt_fg_index] == all_labels_shot[gt_fg_index])[0]) / len(gt_fg_index)

        fg = torch.where(all_labels!=(num_classes-1))[0]

        # if warm==True and len(fg_index)>2:
        #     losses['cont'] = self.con_loss(s_cnn_pol[fg_index],fg_labels,0.07) * weight
        # else:
        # losses['iid'] = IID_loss(nn.Softmax(dim=1)(s_roih_logits[0]), nn.Softmax(dim=1)(t_roih_logits[0]))

        if len(fg) >2 :
            if proposal_variance > 20:
                losses['cont'] = self.con_loss(s_cnn_pol[gt_fg_index],gt_fg_labels,0.1) 
            else: 
                losses['cont'] = self.con_loss(s_cnn_pol,all_labels,0.1) 

        # if len(gt_fg_index) >2 :
        #     losses['cont'] = self.con_loss(s_cnn_pol[gt_fg_index],gt_fg_labels,0.1) 

        # if len(fg) > 2:
        #     all_fea = torch.cat((s_box_features[fg],t_box_features[fg]),dim=0)
        #     all_labels = torch.cat((all_labels[fg],all_labels[fg]),dim=0)
        #     all_fea_pol = self.fc_pol(F.relu(self.CNN(all_fea.detach())))
        #     losses['cont'] = self.con_loss(all_fea_pol,all_labels,0.1) 

            # losses['cont'] = self.con_loss(clus_fea[gt_fg_index],gt_fg_labels,0.1)

            # losses['cls'] = cross_entropy(s_roih_logits[0],all_labels) * 0.1
        # if len(fg) >2 :
        #     if proposal_variance > 20:
        #         losses['cont'] = self.con_loss(t_box_features.mean(dim=[2, 3])[fg],all_labels[fg],0.07) * weight
        #     else:
        #         losses['cont'] = self.con_loss(t_box_features.mean(dim=[2, 3]),all_labels,0.07) * weight

        # logist_new = self.cls_layer(t_cnn_pol)
        # losses['cls1'] = cross_entropy(logist_new,pse_labels)
        # # losses['cls2'] = cross_entropy(logist_new[bg_index],bg_labels)
        # ot_new = nn.Softmax(dim=1)(logist_new)
        # _, predict = torch.max(ot_new, 1)
        # if warm==False:
        #     losses['cls'] = cross_entropy(s_roih_logits[0],all_labels) * 0.4
        # if len(gt_fg_index) !=0 :
        #     acc_new = len(torch.where(gt_labels[gt_fg_index] == predict[gt_fg_index])[0]) / len(gt_fg_index)

        return losses,cluster_acc,pse_acc,count
    

        # ,cluster_acc ,pse_acc, base_acc,acc_new

    def moving_average(self,window, new_value, alpha=0.1):
        if not window:
            window.append(new_value)
            return new_value
        else:
            smoothed_value = alpha * new_value + (1 - alpha) * window[-1]
            window.append(smoothed_value)
            return smoothed_value
        
    def update_sliding_window(self, window, value, window_size):
        if len(window) < window_size:
            window.append(value)
            return np.mean(window), np.std(window)
        else:
            mean = np.mean(window)
            std = np.std(window)
            if not self.is_outlier(value, mean, std):
                window.append(value)
            return np.mean(window), np.std(window)
        
    def is_outlier(self,value, mean, std, threshold=0.2):
        return value > mean + threshold * std
    
    def calculate_proposal_variance(self,proposals, objectness_scores, iou_thresholds):
        proposal_counts = []
        for iou_threshold in iou_thresholds:
            keep = nms(proposals, objectness_scores, iou_threshold)
            proposal_counts.append(len(keep))
        proposal_variance = np.var(proposal_counts)
        return proposal_variance

    def save_step_data(self,t_box_features, gt_labels,  dir):
        print(f"Writing data for step ")
        with open(os.path.join(dir, 'tsne.pkl'), 'ab') as f:
            pickle.dump({'features': t_box_features.detach().cpu().numpy(), 
                        # 'features_pol': t_cnn_pol.detach().cpu().numpy(), 
                        'labels': gt_labels.detach().cpu().numpy()},f)
            
        print(f"Finished writing data for step ")

    def calculate_entropy(self,prob_dist):
        # 为避免log(0)的问题，将概率中的0替换为一个很小的值
        prob_dist = torch.clamp(prob_dist, min=1e-10)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist), dim=1)
        return entropy
    
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

################

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return student_sfda_RCNN_mt_my._postprocess(results, batched_inputs, images.image_sizes)
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
        return images
    
    def preprocess_image_weak(self, batched_inputs: List[Dict[str, torch.Tensor]], mode = "test"):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image_weak"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
    
    def obtain_label(self,all_fea,aff,K,step,fg_index,fg_labels,base_bg_select=None):#获取对应伪标签

        all_output = nn.Softmax(dim=1)(aff)
        _, predict = torch.max(all_output, 1)
        all_fea = torch.cat((all_fea.cpu(), torch.ones(all_fea.size(0), 1)), 1)#加一列 256 -> 257
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()#norm() 范式 p=2 二范式 4365*257 / 1*257 广播机制 ，按列求2范式 （类似softmax）

        all_fea = all_fea.float().cpu().detach().numpy()
        aff = all_output.float().cpu().detach().numpy()
        predict = predict.cpu()
        fg_index = fg_index.cpu()
        fg_labels = fg_labels.cpu()
        # base_bg_select = base_bg_select.cpu()

        for i in range(step):#第一次聚类用net提取的output聚类
            initc = aff.transpose().dot(all_fea) #分类后输出*f 
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count>0)
            # labelset = np.arange(9)

            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], 'cosine')
    
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]
            predict[fg_index]=fg_labels
            # predict[base_bg_select]= self.num_class

            aff = np.eye(K)[predict]

        d = dd
        l = labelset
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]
        label = predict.astype('int')
        return d , l , label


    def pro_obtain_label(self, all_fea, aff, K, prototype, step,fg_index,fg_labels):  # 获取对应伪标签
        act=False

        pro_index = torch.where(self.init == 0)[0]
        if len(pro_index) == 8:
            act = True

        all_output = nn.Softmax(dim=1)(aff)  # Softmax 操作
        _, predict = torch.max(all_output, 1)  # 获取预测标签

        all_fea = torch.cat((prototype, all_fea), dim=0)
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1, device=all_fea.device)), 1)  # 加一列
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()  # 归一化
        
        # 转换为 numpy 数组（如果需要在 CPU 上处理）
        all_fea = all_fea
        aff = all_output
        predict = predict
        pro_index = pro_index
        
        prototype = all_fea[:K-1]
        all_fea = all_fea[K-1:]

        for i in range(step):
            # 使用 PyTorch 进行矩阵运算代替 numpy
            initc = torch.matmul(aff.T, torch.tensor(all_fea, device=aff.device))  # aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(dim=0)[:,None])  # aff.sum(axis=0)[:, None]
            
            if i == 0 and act == True:
                initc[:K-1] = torch.tensor(prototype, device=initc.device)  # prototype 作为 initc 的前 K-1 个
            # else:
            # 计算 cls_count
            cls_count = torch.eye(K, device=initc.device)[predict].sum(dim=0)  # np.eye(K)[predict].sum(axis=0)
            
            # 选取 labelset
            labelset = torch.where(cls_count > 0)[0]

            # 计算 cosine distance（使用 PyTorch 计算余弦距离）
            dot_product = torch.matmul(torch.tensor(all_fea, device=initc.device), initc[labelset].T)  # dot product
            norm_all_fea = torch.norm(torch.tensor(all_fea, device=initc.device), p=2, dim=1, keepdim=True)  # L2 norm
            norm_initc = torch.norm(initc[labelset], p=2, dim=1, keepdim=True)  # L2 norm
            cosine_similarity = dot_product / (norm_all_fea * norm_initc.T)
            
            # 计算 cosine distance
            dd = 1 - cosine_similarity  # cosine distance
            
            # 获取 pred_label
            pred_label = torch.argmin(dd, dim=1)  # 找到每一行的最小值索引
            predict = labelset[pred_label]
            predict[fg_index] = fg_labels

            # 更新 aff
            aff = torch.eye(K, device=initc.device)[predict]
        # 最终返回结果
        d = dd  # 将 PyTorch 张量转回 numpy
        l = labelset  # 将 PyTorch 张量转回 numpy
        pred_label = dd.argmin(axis=1)  # 获取预测标签
        predict = labelset[pred_label]
        label = predict  # 转换为整数标签

        softmax_output = torch.softmax(cosine_similarity / 0.01, dim=1)

        return d, l, label ,softmax_output

    # def pro_obtain_label(self,all_fea,aff,K,prototype,step,pro_index,fg_index,fg_labels):#获取对应伪标签
    #     act=False
    #     if len(pro_index)==8:
    #         act = True
    #     # pro_index = torch.where(self.init==0)[0]
    #     all_output = nn.Softmax(dim=1)(aff)
    #     _, predict = torch.max(all_output, 1)
    #     all_fea = torch.cat((prototype,all_fea),dim=0)
    #     all_fea = torch.cat((all_fea.cpu(), torch.ones(all_fea.size(0), 1)), 1)#加一列 256 -> 257
    #     all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()#norm() 范式 p=2 二范式 4365*257 / 1*257 广播机制 ，按列求2范式 （类似softmax）

    #     all_fea = all_fea.float().cpu().detach().numpy()
    #     aff = all_output.float().cpu().detach().numpy()
    #     predict = predict.cpu()
    #     pro_index = pro_index.cpu()
    #     fg_index = fg_index.cpu()
    #     fg_labels = fg_labels.cpu()
    #     # base_bg_select = base_bg_select.cpu()

    #     prototype = all_fea[:K-1]
    #     all_fea = all_fea[K-1:]
        
    #     for i in range(step):
    #         initc = aff.transpose().dot(all_fea)
    #         initc = initc / (1e-8 + aff.sum(axis=0)[:,None]) 
    #         if i==0 and act==True:
    #             initc[:K-1] = prototype
    #         # if i==0 :
    #         #     initc[pro_index] = prototype[pro_index]
    #         cls_count = np.eye(K)[predict].sum(axis=0)
    #         labelset = np.where(cls_count>0)
    #         labelset = labelset[0]

    #         dd = cdist(all_fea, initc[labelset], 'cosine')

    #         pred_label = dd.argmin(axis=1)
    #         predict = labelset[pred_label]
    #         predict[fg_index]=fg_labels
    #         # predict[base_bg_select]=self.num_class
    #         aff = np.eye(K)[predict]

    #     d = dd
    #     l = labelset
    #     pred_label = dd.argmin(axis=1)
    #     predict = labelset[pred_label]
    #     label = predict.astype('int')

    #     return d , l , label
    

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


def visualize_proposals(cfg, batched_inputs, proposals, index, proposal_dir, metadata,aug):
        from detectron2.utils.visualizer import Visualizer

        for input, prop in zip(batched_inputs, proposals):
            if aug =='w':
                img = input['image_weak']
            if aug =='s':  
                img = input['image_strong']
            img = convert_image_to_rgb(img.permute(1, 2, 0), None)
            # v_gt = Visualizer(img, metadata)
            # v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            # anno_img = v_gt.get_image()
            v_pred = Visualizer(img, metadata)
            if proposal_dir == "rpn":
                v_pred = v_pred.overlay_instances( boxes=prop.proposal_boxes[index].tensor.cpu().numpy())
            if proposal_dir == "pesudo":
                v_pred = v_pred.overlay_instances( boxes=prop.gt_boxes[index].tensor.cpu().numpy())
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

def show_image(cfg, batched_inputs, proposals, index, proposal_dir, metadata,aug):
        from detectron2.utils.visualizer import Visualizer

        for input, prop in zip(batched_inputs, proposals):
            if aug =='s':
                img = input['image_weak']
            if aug =='w':  
                img = input['image_strong']
            img = convert_image_to_rgb(img.permute(1, 2, 0), None)
            # v_gt = Visualizer(img, metadata)
            # v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            # anno_img = v_gt.get_image()
            v_pred = Visualizer(img, metadata)
            if proposal_dir == "rpn":
                v_pred = v_pred.overlay_instances( boxes=prop.proposal_boxes[index].tensor.cpu().numpy())
            if proposal_dir == "pesudo":
                v_pred = v_pred.overlay_instances( boxes=prop.gt_boxes[index].tensor.cpu().numpy())
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

def visualize_map(res5,images):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    # 假设 feature_map 是你的 [1, 2048, 19, 38] 特征图 (PyTorch Tensor)
    feature_map = res5  # 示例特征图
    feature_map = feature_map.squeeze(0).detach().cpu()  # 移除 batch 维度，结果是 [2048, 19, 38]

    # 加载原始 RGB 图像 (PyTorch Tensor)
    original_image = images.squeeze(0).detach().cpu()
    original_image = original_image.permute(1, 2, 0).numpy()  # 转换为 [H, W, C]
    original_image = (original_image * 255 / original_image.max()).astype(np.uint8)  # 转换为 uint8 类型

    # 从特征图中选择一个通道进行可视化
    aggregated_map = torch.mean(feature_map, dim=0).numpy()  # 结果是 [19, 38]

    # 将特征图 resize 到与原始图像相同的大小
    h, w, _ = original_image.shape
    resized_feature = cv2.resize(aggregated_map, (w, h))

    # 归一化特征图到 0-255
    resized_feature = cv2.normalize(resized_feature, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 将热图应用颜色映射
    heatmap = cv2.applyColorMap(resized_feature, cv2.COLORMAP_JET)

    # 创建一个叠加图像
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    # 将热图叠加到原始图像上
    overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)

    # 可视化结果
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title("Feature Map Overlay on Original Image")
    plt.axis('off')
    plt.show()

def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    original_indices = filter_inds[:, 0]

    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.

    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.idx = original_indices[keep]

    return result,

def process_pseudo_label(proposals_rpn_k, cur_threshold, proposal_type, psedo_label_method=""):
    list_instances = []
    num_proposal_output = 0.0

    for proposal_bbox_inst in proposals_rpn_k:
        # thresholding
        if psedo_label_method == "thresholding":
            proposal_bbox_inst = threshold_bbox(
                proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
            )
        else:
            raise ValueError("Unkown pseudo label boxes methods")
        num_proposal_output += len(proposal_bbox_inst)
        list_instances.append(proposal_bbox_inst)

    num_proposal_output = num_proposal_output / len(proposals_rpn_k)
    return list_instances, num_proposal_output

def threshold_bbox(proposal_bbox_inst, thres=0.7, proposal_type="roih"):
    if proposal_type == "rpn":
        valid_map = proposal_bbox_inst.objectness_logits > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
            valid_map
        ]
    elif proposal_type == "roih":
        valid_map = proposal_bbox_inst.scores > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
        new_proposal_inst.idx = proposal_bbox_inst.idx[valid_map]

    return new_proposal_inst