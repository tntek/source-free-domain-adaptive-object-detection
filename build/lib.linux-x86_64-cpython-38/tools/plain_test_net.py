#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.structures import Boxes, pairwise_iou ,ImageList
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    ClipartDetectionEvaluator,
    CityscapeDetectionEvaluator,
    FoggyDetectionEvaluator,
    Sim10kDetectionEvaluator,
    CityscapeCarDetectionEvaluator,
    WatercolorDetectionEvaluator
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from pycocotools.cocoeval import COCOeval

import pdb

logger = logging.getLogger("detectron2")
import xml.etree.ElementTree as ET
import torch

def parse_annotations(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    return torch.tensor(boxes)

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name,cfg)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name,cfg)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if evaluator_type == "clipart":
        return ClipartDetectionEvaluator(dataset_name,cfg)
    if evaluator_type == "cityscape":
        return CityscapeDetectionEvaluator(dataset_name,cfg)
    if evaluator_type == "foggy":
        return FoggyDetectionEvaluator(dataset_name,cfg)
    if evaluator_type == "sim10k":
        return Sim10kDetectionEvaluator(dataset_name,cfg)
    if evaluator_type == "cityscape_car":
        return CityscapeCarDetectionEvaluator(dataset_name,cfg)
    if evaluator_type == "watercolor":
        return WatercolorDetectionEvaluator(dataset_name,cfg)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


# def do_test(cfg, model):
#     results = OrderedDict()
#     for dataset_name in cfg.DATASETS.TEST:
#         data_loader = build_detection_test_loader(cfg, dataset_name)
#         test_metadata = MetadataCatalog.get(dataset_name)

#         evaluator = get_evaluator(
#             cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
#         )
#         results_i = inference_on_dataset(model, data_loader, evaluator)
#         results[dataset_name] = results_i
#         # if comm.is_main_process():
#         logger.info("Evaluation results for {} in csv format:".format(dataset_name))
#         print_csv_format(results_i)
#         cls_names = test_metadata.get("thing_classes")
#         cls_aps = results_i['bbox']['class-AP50']
#         for i in range(len(cls_aps)):
#             logger.info("AP for {}: {}".format(cls_names[i], cls_aps[i]))
#     if len(results) == 1:
#         results = list(results.values())[0]
#     return results

def resize_boxes(boxes, original_size, new_size):
    """ Rescale the boxes from original_size to new_size """
    ratios = [float(new_size[0]) / original_size[0], float(new_size[1]) / original_size[1]]
    resized_boxes = boxes.clone()
    resized_boxes[:, [0, 2]] *= ratios[1]  # x scaling
    resized_boxes[:, [1, 3]] *= ratios[0]  # y scaling
    return resized_boxes

def do_test(cfg, model):
    results = OrderedDict()
    # model.eval()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        test_metadata = MetadataCatalog.get(dataset_name)
        
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        # evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/")

        results[dataset_name] = results_i


        # evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/")

        # 如果comm.is_main_process():
        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_csv_format(results_i)
        cls_names = test_metadata.get("thing_classes")
        cls_aps = results_i['bbox']['class-AP50']
        for i in range(len(cls_aps)):
            logger.info('AP for {}: {}'.format(cls_names[i], cls_aps[i]))
    
    if len(results) == 1:
        results = list(results.values())[0]
    
    return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file('/home/eason/AAAI/aa/analysis/sfda_watercolor.yaml')
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR+'/' + args.where
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    # if args.eval_only:
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load('/home/eason/AAAI/aa/best_model_water_56.9.pth')
    logger.info("Trained model has been sucessfully loaded")
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    from datetime import datetime
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d-%H:%M:%S")

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
