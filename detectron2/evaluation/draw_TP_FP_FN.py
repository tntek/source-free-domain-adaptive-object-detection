import os
import torch

def IoU_boxes(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0]-torch.tensor(w1)/2.0, boxes2[0]-torch.tensor(w2)/2.0)
        x2_max = torch.max(boxes1[0]+torch.tensor(w1)/2.0, boxes2[0]+torch.tensor(w2)/2.0)
        y1_min = torch.min(boxes1[1]-torch.tensor(h1)/2.0, boxes2[1]-torch.tensor(h2)/2.0)
        y2_max = torch.max(boxes1[1]+torch.tensor(h1)/2.0, boxes2[1]+torch.tensor(h2)/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea / uarea)


import xml.etree.ElementTree as ET
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        # 匹配数据集
        try: obj_struct["pose"] = obj.find("pose").text
        except: obj_struct["pose"] = 'Frontal'
        try: obj_struct["truncated"] = int(obj.find("truncated").text) 
        except: obj_struct["truncated"] = 0
        try: obj_struct["difficult"] = int(obj.find("difficult").text)
        except: obj_struct["difficult"] = 0
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects

def compute(new_truths, detect_boxes, iou_thresh):
    TP_boxes = []
    falsenegative = []
    FN,TP0 = 0, 0
    matched_gts_num = 0
    for box_i in new_truths:
        check_TP = False
        for box_j in detect_boxes:
            if IoU_boxes(box_i, box_j, x1y1x2y2=True) >= iou_thresh:
                TP0 += 1
                check_TP = True
                TP_boxes.append(box_j)
        if check_TP:
            matched_gts_num += 1
        if not check_TP:
            # print(imgfile, ' ', IoU_boxes(box_i, box_j, x1y1x2y2=False))
            falsenegative.append(box_i)
            FN += 1

    false_positive = []
    FP, TP1 = 0, 0
    for box_i in detect_boxes:
        # print(box)
        check_TP = False
        for box_j in new_truths:
            if IoU_boxes(box_i, box_j, x1y1x2y2=True) >= iou_thresh :
            # if IoU_boxes(box_i, box_j, x1y1x2y2=True) > 0 :
                TP1 += 1
                check_TP = True
        if not check_TP:
            # print(imgfile,' ',IoU_boxes(box_i, box_j, x1y1x2y2=False))
            false_positive.append(box_i)
            FP += 1
    
    assert TP0 == TP1
    # assert FP == len(detect_boxes) - TP0
    assert FN == len(new_truths) -  matched_gts_num
    # print('True Positive = %d \t False Positive = %d \t False Negative = %d \n', TP,FP,FN)

    # for box_i in falsenegative:
    #     # print(box)
    #     for box_j in false_positive:
    #         if IoU_boxes(box_i, box_j, x1y1x2y2=False) >= iou_thresh:
    #             FN -= 1
    #             falsenegative.remove(box_i)
    #             false_positive.remove(box_j)
    #             TP_boxes.append(box_j)
    # print(len(detect_boxes),TP0,FP)
    return TP_boxes, false_positive, falsenegative


# def compute(new_truths, detect_boxes, iou_thresh):
#     TP_boxes = []
#     false_positive = []
#     falsenegative = []

#     TP, FP, FN = 0, 0, 0
#     matched_detect_indices = set()  # 用来记录已经匹配的检测框索引
#     matched_gts_num = 0

#     # 遍历 ground truth 来计算 TP 和 FN
#     for gt_idx, box_i in enumerate(new_truths):
#         check_TP = False
#         for det_idx, box_j in enumerate(detect_boxes):
#             if IoU_boxes(box_i, box_j, x1y1x2y2=True) >= iou_thresh:
#                 # 只要当前检测框还没有被匹配过，就计为 TP
#                 if det_idx not in matched_detect_indices:
#                     TP += 1
#                     matched_detect_indices.add(det_idx)  # 标记该检测框已匹配
#                     TP_boxes.append(box_j)
#                     check_TP = True
#                     matched_gts_num += 1
#                     break  # 当前目标框已经匹配到检测框，跳出检测框循环
#         if not check_TP:
#             FN += 1
#             falsenegative.append(box_i)  # 如果目标框没有匹配到任何检测框，计为 FN

#     # 遍历检测框来计算 FP
#     for det_idx, box_i in enumerate(detect_boxes):
#         if det_idx not in matched_detect_indices:  # 如果当前检测框未匹配
#             FP += 1
#             false_positive.append(box_i)

#     return TP_boxes, false_positive, falsenegative




import math
def plot_boxes_cv2(img, boxes, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    # print("%d box(es) is(are) found" % len(boxes))
    for i in range(len(boxes)):
        box = boxes[i]
        box = box.tolist()
        x1 = int(round(box[0]))
        y1 = int(round(box[1]))
        x2 = int(round(box[2]))
        y2 = int(round(box[3]))
        # x1 = int(round((box[0] - box[2]/2.0) * width))
        # y1 = int(round((box[1] - box[3]/2.0) * height))
        # x2 = int(round((box[0] + box[2]/2.0) * width))
        # y2 = int(round((box[1] + box[3]/2.0) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = float(box[5])
            # print('conf before: ',cls_conf)
            # cls_conf = round(cls_conf,2)
            # print('conf after: ',cls_conf)
            cls_id = int(box[6])
            #print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            # img = cv2.putText(img, class_names[cls_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rgb, 1)
            ###this is for plot the score
            # img = cv2.putText(img, str(cls_conf), (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rgb, 1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 2)
    
    return img

import numpy as np
def load_ananos(imagenames, classname, recs):
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = torch.FloatTensor([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    return class_recs


import cv2
import argparse
import pickle
from detectron2.utils.file_io import PathManager

def count_martix(dets_path,score_thresh,img_prefex):

    """
    Parse input arguments
    """
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)


    f = open(dets_path, 'rb')
    obj = pickle.load(f)
    f.close()
    dets, gt_path, annopath, class_names = obj

    # 处理检测结果
    detections = []
    for c in sorted(dets.keys()):
        detections.append({})
        lines = dets[c]
        splitlines = [x.strip().split(" ") for x in lines]
        image_ids = set([x[0] for x in splitlines])
        for img in image_ids:
            result = torch.FloatTensor([[float(z) for z in x[2:]] + [float(x[1])] for x in splitlines if x[0]==img])
            detections[c][img] = result

    with PathManager.open(gt_path, "r") as f:
        lines = f.readlines()
    file_names = [x.strip() for x in lines]

    recs = {}
    for file_name in file_names:
        recs[file_name] = parse_rec(annopath.format(file_name))

    img_temp = annopath.replace('Annotations', 'JPEGImages')
    img_temp = img_temp.replace('xml', img_prefex)

    # 我的逐个类别的gts
    gts = [load_ananos(file_names, name, recs) for name in class_names]
    all_TPS ,all_FPS ,all_FNS = 0,0,0

    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)

    all_gt = []
    all_pred = []

    # 对每个文件进行遍历
    for file_name in file_names:
        img_path = img_temp.format(file_name)
        img = cv2.imread(img_path)

        total_TPS ,total_FPS ,total_FNS = 0,0,0

        for c in range(len(class_names)):
            try:
                det_c = detections[c][file_name]
            except:
                det_c = torch.empty(0, 5)
            gt_c = gts[c][file_name]

            det_c = [x[:-1] for x in det_c if x[-1] >= score_thresh]
            gt_c = [x for x in gt_c['bbox']]
            
            TPS, FPS, FNS = compute(gt_c, det_c, iou_thresh=0.5)

            # 更新混淆矩阵
            for pred_box in TPS:
                pred_class = int(pred_box[-1])  # 预测框的类别
                for gt_box in gt_c:
                    gt_class = gt_box['class']  # 真实框的类别
                    if pred_class == gt_class:
                        confusion_matrix[pred_class, gt_class] += 1

            for pred_box in FPS:
                pred_class = int(pred_box[-1])  # 预测框的类别
                confusion_matrix[pred_class, -1] += 1  # 错误预测框的背景（未分类）

            for gt_box in FNS:
                gt_class = gt_box['class']  # 真实框的类别
                confusion_matrix[-1, gt_class] += 1  # 错误标记为背景的真实框

            img = plot_boxes_cv2(img, FPS, color=blue)
            img = plot_boxes_cv2(img, FNS, color=red)
            img = plot_boxes_cv2(img, TPS, color=green)

    # 输出最终的混淆矩阵
    print("Confusion Matrix: \n", confusion_matrix)


        
    fnr = all_FNS /(all_TPS + all_FNS)
    
    # print(all_TPS,all_FPS,all_FNS,fnr)

    return all_TPS,all_FPS,all_FNS,fnr

    # cv2.imwrite(output_path,img)

def count(dets_path,score_thresh,img_prefex):

    """
    Parse input arguments
    """
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)


    f = open(dets_path, 'rb')
    obj = pickle.load(f)
    f.close()
    dets, gt_path, annopath, class_names = obj

    # 处理检测结果
    detections = []
    for c in sorted(dets.keys()):
        detections.append({})
        lines = dets[c]
        splitlines = [x.strip().split(" ") for x in lines]
        image_ids = set([x[0] for x in splitlines])
        for img in image_ids:
            result = torch.FloatTensor([[float(z) for z in x[2:]] + [float(x[1])] for x in splitlines if x[0]==img])
            detections[c][img] = result

    with PathManager.open(gt_path, "r") as f:
        lines = f.readlines()
    file_names = [x.strip() for x in lines]

    recs = {}
    for file_name in file_names:
        recs[file_name] = parse_rec(annopath.format(file_name))

    img_temp = annopath.replace('Annotations', 'JPEGImages')
    img_temp = img_temp.replace('xml', img_prefex)

    # 我的逐个类别的gts
    gts = [load_ananos(file_names, name, recs) for name in class_names]
    all_TPS ,all_FPS ,all_FNS = 0,0,0

    # 对每个文件进行遍历
    for file_name in file_names:
        img_path = img_temp.format(file_name)
        img = cv2.imread(img_path)

        total_TPS ,total_FPS ,total_FNS = 0,0,0

        for c in range(len(class_names)):
            try:
                det_c = detections[c][file_name]
            except:
                det_c = torch.empty(0, 5)
            gt_c = gts[c][file_name]

            det_c = [x[:-1] for x in det_c if x[-1] >= score_thresh]
            gt_c = [x for x in gt_c['bbox']]
            TPS, FPS, FNS = compute(gt_c, det_c, iou_thresh=0.5)

            total_TPS +=len(TPS)
            total_FPS +=len(FPS)
            total_FNS +=len(FNS)

            all_TPS +=len(TPS)
            all_FPS +=len(FPS)
            all_FNS +=len(FNS)

            img = plot_boxes_cv2(img, FPS, color=blue)
            img = plot_boxes_cv2(img, FNS, color=red)
            img = plot_boxes_cv2(img, TPS, color=green)

        # output_file_name = "{}-{}-{}-{}.{}".format(file_name ,total_TPS ,total_FPS ,total_FNS, args.img_prefex)
        # cv2.imwrite(output_path,img)
        
    fnr = all_FNS /(all_TPS + all_FNS)
    
    # print(all_TPS,all_FPS,all_FNS,fnr)

    return all_TPS,all_FPS,all_FNS,fnr

    # cv2.imwrite(output_path,img)
