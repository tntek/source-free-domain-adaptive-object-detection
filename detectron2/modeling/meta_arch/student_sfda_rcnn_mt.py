# Copyright (c) Facebook, Inc. and its affiliates.
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

__all__ = ["student_sfda_RCNN_mt"]
@META_ARCH_REGISTRY.register()
class student_sfda_RCNN_mt(nn.Module):
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
        self.init = torch.ones(8, device='cuda') #class_n 个 1
        # self.init_pol = torch.ones(8, device='cpu') #class_n 个 1

        self.memory_bank = MemoryBank(8,10)
        # self.classifier = nn.Linear(512,9)
        self.criterion = nn.L1Loss(reduction='none')
        self.prototype = torch.zeros([8,2048],dtype=torch.float,device='cuda')
        self.prototype_pol = torch.zeros([8,512],dtype=torch.float,device='cuda')
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "cfg":cfg,
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
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], cfg=None, model_teacher=None, t_features=None, t_proposals=None, t_results=None,epoch=None,weight=None,emaweight=None,move=None,cluster_step=None,temp=None,mode="test"):

        if not self.training and mode == "test":
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs, mode)
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features,t_results)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        results, detector_losses = self.roi_heads(images, features, proposals,t_results) #>0.5前景

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        s_box_features = self.roi_heads._shared_roi_transform([features['res4']], [t_proposals[0].proposal_boxes]) #t_proposals[0], results[1]
        s_roih_logits = self.roi_heads.box_predictor(s_box_features.mean(dim=[2, 3]))
        # visualize_proposals(cfg,batched_inputs,t_proposals,0,3,'rpn',MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),'s')
        t_box_features = model_teacher.roi_heads._shared_roi_transform([t_features['res4']], [t_proposals[0].proposal_boxes])
        t_roih_logits = model_teacher.roi_heads.box_predictor(t_box_features.mean(dim=[2, 3]))
    
        t_s_roih_logits = self.roi_heads.box_predictor(t_box_features.mean(dim=[2, 3]))

        fg_index,fg_labels,bg_index,bg_label,pse_labels = model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,t_results)

        all_index = torch.cat((fg_index,bg_index),dim=0)

        #memory
        pse_id = torch.where(t_results[0]._fields['scores'])[0]
        pse_f_no_pol =  self.roi_heads._shared_roi_transform([t_features['res4']], [t_results[0]._fields['gt_boxes']])[pse_id].detach()
        # pse_fea =  pse_f_no_pol.mean(dim=[2, 3])
        
        pse_l = t_results[0]._fields['gt_classes'][pse_id].detach()
        # pse_logits = model_teacher.roi_heads.box_predictor(pse_fea)[0].detach()
        # pse_output = nn.Softmax(dim=1)(pse_logits)
        # _, pse_l = torch.max(pse_output, 1)
        self.memory_bank.update(pse_f_no_pol,pse_l)

        me_f,me_l = self.memory_bank.read(10)
        m_f = torch.stack(me_f).cuda()
        m_l = torch.tensor(me_l).cuda()
        m_f_pol = m_f.mean(dim=[2, 3])
        m_logits = model_teacher.roi_heads.box_predictor(m_f_pol)[0].detach()

        losses["cma"] = self.feat_kd_high_loss(features['res4'],t_features['res4']) * 0.001
        losses["st_const"] = self.KD_loss(s_roih_logits[0], t_roih_logits[0]) 

        s_cnn_feat = self.CNN(s_box_features)
        t_cnn_feat = self.CNN(t_box_features).detach()

        s_cnn_pol = self.fc_pol(F.relu(s_cnn_feat))
        t_cnn_pol = self.fc_pol(F.relu(t_cnn_feat)).detach()
        memory_cnn_pol = self.fc_pol(F.relu(self.CNN(m_f))).detach()
        
        # prototype = self.prototype_update(m_f_pol,m_l,emaweight)
        prototype_pol = self.prototype_update_pol(memory_cnn_pol,m_l,emaweight)

        # if len(pro_index)!=0:
        #     m_f_pol = m_f_pol[pro_index]
        #     memory_cnn_pol = memory_cnn_pol[pro_index]
        #     m_logits = m_logits[pro_index]
        #     m_l = m_l[pro_index]

        memory_fea = torch.cat((t_box_features.mean(dim=[2, 3]),m_f_pol),dim=0)
        memory_fea_pol = torch.cat((t_cnn_pol,memory_cnn_pol),dim=0)
        memory_logits = torch.cat((t_roih_logits[0],m_logits),dim=0)
        # memory_logits_s = torch.cat((s_roih_logits[0],m_logits),dim=0)

        pro_index = torch.where(self.init!=1)[0]

        # dd ,labelset,label_pro= self.pro_obtain_label(t_box_features.mean(dim=[2, 3]),t_roih_logits[0],9,prototype,cluster_step,pro_index,act)
        dd_c ,labelset_c,all_labels= self.pro_obtain_label(t_cnn_pol,t_s_roih_logits[0],9,prototype_pol,cluster_step,pro_index)
        # if len(pro_index)==8:
        #     losses["pro_const"] = self.pro_const_loss(s_cnn_pol,t_cnn_pol,self.prototype_pol)

        if epoch==1:
            dd ,labelset_m,label_clus= self.obtain_label(memory_fea,memory_logits,9,2)
            # dd_cnn ,labelset_s_,label_cnn_pro_m= self.pro_obtain_label(memory_fea_pol,memory_logits,9,prototype_pol,cluster_step,pro_index,act)
            # dd,labelset_s,label_shot= self.obtain_label(t_box_features.mean(dim=[2, 3]),t_roih_logits[0],9,2)

            if np.array_equal(labelset_c, labelset_m):
                dd = (1 - move ) * dd[:len(s_box_features)] + ( move ) * dd_c

            pred_label = dd.argmin(axis=1)
            predict = labelset_m[pred_label]
            all_labels = predict.astype('int')

        base_output = nn.Softmax(dim=1)(t_s_roih_logits[0])
        _, base_l = torch.max(base_output, 1)

        base_bg_index = torch.where(base_l==8)[0]
        base_bg_logtis_top2,_ = torch.topk(base_output[base_bg_index],k=2)

        base_bg_select = base_bg_index[torch.where(base_bg_logtis_top2[:,0] > 0.95)]
        all_labels = torch.tensor(all_labels[:len(t_box_features)]).cuda()
        # all_labels[base_bg_select] = 8
        all_labels[fg_index] = fg_labels

        # all_labels_shot = torch.tensor(label_shot[:len(s_box_features)]).cuda()
        # all_labels_shot[fg_index] = fg_labels
        
        fg = torch.where(all_labels!=8)[0]
        gt_fg_index,gt_fg_labels,gt_bg_index,gt_bg_labels,gt_labels= model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,gt_instances)

        if len(fg) >2 :

            # st_cnn_pol = torch.cat((s_cnn_pol[fg],t_cnn_pol[fg]),dim=0)
            # labels = torch.cat((all_labels[fg],all_labels[fg]))

            losses['cont'] = self.con_loss(s_cnn_pol[fg],all_labels[fg],temp) * weight

        # model_teacher.roi_heads.proposal_matcher.thresholds[1] = 0.5
        gt_fg_index,gt_fg_labels,gt_bg_index,gt_bg_labels,gt_labels= model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,gt_instances)
        p_fg_index,p_fg_labels,_,_,p_labels= model_teacher.roi_heads.label_and_sample_proposals_withoutgt(t_proposals,t_results)

        # model_teacher.roi_heads.proposal_matcher.thresholds[1] = 0.5

        cluster_acc ,cnn_acc_pro,base_acc,shot_acc=0,0,0,0
        pse_acc = 0
        if len(gt_fg_index) !=0 :

            cluster_acc = len(torch.where(gt_labels[gt_fg_index] == all_labels[gt_fg_index])[0]) / len(gt_fg_index)
            # # pro_acc = len(torch.where(gt_labels == all_labels_pro)[0]) / len(pse_labels)
            # # cnn_acc = len(torch.where(gt_labels == all_labels_cnn)[0]) / len(pse_labels)
            # cnn_acc_pro = len(torch.where(gt_labels[gt_fg_index] == all_labels_cnn_pro[gt_fg_index])[0]) / len(gt_fg_index)
            # # mix_acc = len(torch.where(gt_labels == mix_label)[0]) / len(pse_labels)
            # # cnn_pro_m = len(torch.where(gt_labels == label_cnn_pro_m)[0]) / len(pse_labels)
            pse_acc = len(torch.where(gt_labels[gt_fg_index] == p_labels[gt_fg_index])[0]) / len(gt_fg_index)
            # # shot_acc = len(torch.where(gt_labels[gt_fg_index] == all_labels_shot[gt_fg_index])[0]) / len(gt_fg_index)
            base_acc = len(torch.where(gt_labels[gt_fg_index] == base_l[gt_fg_index])[0]) / len(gt_fg_index)

        return losses , cluster_acc ,pse_acc, base_acc

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
            return student_sfda_RCNN_mt._postprocess(results, batched_inputs, images.image_sizes)
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
    
    def obtain_label(self,all_fea,aff,K,step,prototype=None):#获取对应伪标签

        all_output = nn.Softmax(dim=1)(aff)
        _, predict = torch.max(all_output, 1)
        all_fea = torch.cat((all_fea.cpu(), torch.ones(all_fea.size(0), 1)), 1)#加一列 256 -> 257
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()#norm() 范式 p=2 二范式 4365*257 / 1*257 广播机制 ，按列求2范式 （类似softmax）

        all_fea = all_fea.float().cpu().detach().numpy()
        aff = all_output.float().cpu().detach().numpy()
        predict = predict.cpu()

        for i in range(step):#第一次聚类用net提取的output聚类
            initc = aff.transpose().dot(all_fea) #分类后输出*f 
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])

            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count>0)
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], 'cosine')
    
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]

            aff = np.eye(K)[predict]

        d = dd
        l = labelset
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]
        label = predict.astype('int')
        return d , l , label

    def pro_obtain_label(self,all_fea,aff,K,prototype,step,pro_index,act=False):#获取对应伪标签
        if len(pro_index)==8:
            act = True
    
        all_output = nn.Softmax(dim=1)(aff)
        _, predict = torch.max(all_output, 1)
        all_fea = torch.cat((prototype,all_fea),dim=0)
        all_fea = torch.cat((all_fea.cpu(), torch.ones(all_fea.size(0), 1)), 1)#加一列 256 -> 257
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()#norm() 范式 p=2 二范式 4365*257 / 1*257 广播机制 ，按列求2范式 （类似softmax）

        all_fea = all_fea.float().cpu().detach().numpy()
        aff = all_output.float().cpu().detach().numpy()
        predict = predict.cpu()
        pro_index = pro_index.cpu()

        prototype = all_fea[:K-1]
        all_fea = all_fea[K-1:]
        
        for i in range(step):
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None]) 
            if i==0 and act==True:
                
                initc[:K-1] = prototype
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count>0)
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], 'cosine')

            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]
            
            aff = np.eye(K)[predict]


        d = dd
        l = labelset
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]
        label = predict.astype('int')
        return d , l , label
    
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



def visualize_proposals(cfg, batched_inputs, proposals, start,end, proposal_dir, metadata,aug):
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

