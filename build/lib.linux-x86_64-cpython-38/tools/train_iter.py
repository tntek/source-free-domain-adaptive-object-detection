import logging
import os
import copy
import torch.optim as optim
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
from detectron2.data.masking import Masking

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
    WatercolorDetectionEvaluator,
    CityscapeDetectionEvaluator,
    FoggyDetectionEvaluator,
    CityscapeCarDetectionEvaluator,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

import pdb
import cv2
from pynvml import *
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.data.detection_utils import convert_image_to_rgb

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger("detectron2")


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
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if evaluator_type == "clipart":
        return ClipartDetectionEvaluator(dataset_name)
    if evaluator_type == "watercolor":
        return WatercolorDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape":
        return CityscapeDetectionEvaluator(dataset_name)
    if evaluator_type == "foggy":
        return FoggyDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape_car":
        return CityscapeCarDetectionEvaluator(dataset_name)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

# =====================================================
# ================== Pseduo-labeling ==================
# =====================================================
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
        new_proposal_inst.idx = torch.where(valid_map)[0]


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

def process_pseudo_label_water(proposals_rpn_k, cur_threshold, proposal_type, psedo_label_method=""):
    class_index = torch.tensor([1,2,6,7,11,14]).cuda()
    list_instances = []
    num_proposal_output = 0.0
    for proposal_bbox_inst in proposals_rpn_k:
        # thresholding
        mask = torch.isin(proposal_bbox_inst._fields['pred_classes'],class_index)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        if psedo_label_method == "thresholding":
            proposal_bbox_inst = threshold_bbox(
                proposal_bbox_inst[indices], thres=cur_threshold, proposal_type=proposal_type
            )
        else:
            raise ValueError("Unkown pseudo label boxes methods")
        
        num_proposal_output += len(proposal_bbox_inst)
        list_instances.append(proposal_bbox_inst)

    num_proposal_output = num_proposal_output / len(proposals_rpn_k)
    return list_instances, num_proposal_output

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.996):
    if comm.get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model_student.state_dict().items()
        }
    else:
        student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    return new_teacher_dict

@torch.no_grad()
def update_student_model(model_student, model_teacher, keep_rate=0.996):
    if comm.get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model_student.state_dict().items()
        }
    else:
        student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            new_teacher_dict[key] = value

    return new_teacher_dict

def visualize_proposals(cfg, batched_inputs, proposals, box_size, proposal_dir, metadata):
        from detectron2.utils.visualizer import Visualizer

        for input, prop in zip(batched_inputs, proposals):
            img = input["image_weak"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), None)
            #v_gt = Visualizer(img, None)
            #v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            #anno_img = v_gt.get_image()
            v_pred = Visualizer(img, metadata)
            if proposal_dir == "rpn":
                v_pred = v_pred.overlay_instances( boxes=prop.proposal_boxes[0:int(box_size)].tensor.cpu().numpy())
            if proposal_dir == "roih":
                v_pred = v_pred.draw_instance_predictions(prop)
            vis_img = v_pred.get_image()

            save_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir) 
            save_img_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir, input['file_name'].split('/')[-1]) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_img_path, vis_img)


def test_sfda(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        cfg.defrost()
        cfg.SOURCE_FREE.TYPE = False
        # cfg.MODEL.RPN.POST_NMS_TOPK_TEST: 300
        cfg.freeze()
        # model.proposal_generator.post_nms_topk[False] = 1000
        test_data_loader = build_detection_test_loader(cfg, dataset_name)
        test_metadata = MetadataCatalog.get(dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, test_data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
            #pdb.set_trace()
            cls_names = test_metadata.get("thing_classes")
            cls_aps = results_i['bbox']['class-AP50']
            for i in range(len(cls_aps)):
                logger.info("AP for {}: {}".format(cls_names[i], cls_aps[i]))
    if len(results) == 1:
        results = list(results.values())[0]

    cfg.defrost()
    cfg.SOURCE_FREE.TYPE = True
    cfg.freeze()

    return results


def train_sfda(cfg, model_student, model_teacher,model_teacher_s):
    
    checkpoint = copy.deepcopy(model_teacher.state_dict())

    model_teacher.eval()
    model_teacher_s.eval()
    model_student.train()

    #optimizer = optim.SGD(model_student.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    optimizer = build_optimizer(cfg, model_student)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(model_student, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    #pdb.set_trace()

    data_loader = build_detection_train_loader(cfg)

    total_epochs = 5
    len_data_loader = len(data_loader.dataset.dataset.dataset)
    start_iter, max_iter = 0, len_data_loader
    max_sf_da_iter = total_epochs*max_iter
    logger.info("Starting training from iteration {}".format(start_iter))

    # periodic_checkpointer = PeriodicCheckpointer(checkpointer, len_data_loader, max_iter=max_sf_da_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_sf_da_iter) if comm.is_main_process() else []

    # model_teacher.eval()
    # test_sfda(cfg, model_teacher)

    with EventStorage(start_iter) as storage:
        
        # for epoch in range(1, total_epochs+1):
        #     cfg.defrost()
        #     cfg.SOURCE_FREE.TYPE = True
        #     cfg.freeze()
        #     model_teacher.eval()
        #     model_student.train()
        #     # model_student.reset_memory_bank()

            a,b,c,d = 0,0,0,0
        #     # d,e,f,g = 0,0,0,0
            a1,b1,c1 ,d1= 0,0,0,0
        #     # a2,b2,c2 ,d1= 0,0,0,0

        #     r1 = 0
            # start_time = time.time()

            data_loader = build_detection_train_loader(cfg)
            # masking = Masking(block_size=64, masked_ratio=0.5)

            for data, iteration in zip(data_loader, range(start_iter, max_sf_da_iter)):

                # batch_data, indices = data
                # batch_data = [batch_data]

                storage.iter = iteration
                epoch = (iteration // max_iter)+1


                model_teacher.eval()
                model_teacher_s.eval()
                model_student.train()

                with torch.no_grad():
                    # model_teacher.proposal_generator.post_nms_topk[False] = 300
                    # _, teacher_features, teacher_proposals, teacher_results = model_teacher(data, mode="train")
                    _, teacher_features, teacher_proposals, teacher_results = model_teacher(data, mode="train")

                    # model_teacher.proposal_generator.post_nms_topk[False] = 1000
                    # _, teacher_features_1k, teacher_proposals_1k, teacher_results_1k = model_teacher(data, mode="train")

                # teacher_pseudo_proposals, num_rpn_proposal = process_pseudo_label(teacher_proposals, 0.9, "rpn", "thresholding")
                teacher_pseudo_results_rpn, num_roih_proposal_rpn = process_pseudo_label(teacher_results, 0.9, "roih", "thresholding")
                # teacher_pseudo_results, num_roih_proposal = process_pseudo_label(teacher_results, 0.7, "roih", "thresholding")
                teacher_pseudo_results, num_roih_proposal = process_pseudo_label(teacher_results, 0.7, "roih", "thresholding")
                
                if len(teacher_pseudo_results[0])==0:
                    continue

                if iteration <= 500:
                    warm = True
                else:
                    warm = False

                move = iteration / 500
                loss_dict,cluster_acc,pse_acc,count = model_student(data, cfg, model_teacher, teacher_features, teacher_proposals, teacher_pseudo_results,teacher_pseudo_results_rpn,epoch,args.cmaweight,args.emaweight,move,args.cluster_step,args.temp,warm,mode="train")

                a +=cluster_acc /50
                b +=pse_acc /50
                c +=count
                # d +=acc_new /50

                # # d += cluster_acc /50
                # e +=pse_acc /50
                # f +=base_acc /50
                # g +=acc_new /50

                a1 +=cluster_acc /200
                b1 +=pse_acc /200
                # c1 +=base_acc /max_iter
                # d1 +=acc_new /max_iter

                # a2 += cluster_acc /2965
                # b2 +=pse_acc /2965
                # c2 +=base_acc /2965

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                scheduler.step()

                new_teacher_dict = update_teacher_model(model_student, model_teacher, keep_rate= 0.9996)
                model_teacher.load_state_dict(new_teacher_dict)

                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                # if iteration - start_iter > 5 and ((iteration + 1) % 50 == 0 or iteration == max_iter - 1):
                #     print("epoch: ", epoch, "lr:", optimizer.param_groups[0]["lr"], ''.join(['{0}: {1}, '.format(k, v.item()) for k,v in loss_dict.items()]))
                # if comm.is_main_process():
                #     for writer in writers:
                #         writer.write()
                if iteration - start_iter > 5 and ((iteration + 1) % 50 == 0 or iteration == max_iter - 1):
                    # print("epoch: ", epoch, "lr:", optimizer.param_groups[0]["lr"], "pse_acc:",pse_acc, "cluter_acc:",cluster_acc ,''.join(['{0}: {1}, '.format(k, v.item()) for k,v in loss_dict.items()]))
                    r = a-b
                    logger.info("epoch:{}, iter:{}, cluster acc: {:.3f},pse_acc: {:.3f},pse_acc: {:.3f},loss :{:}".format(epoch,iteration,a,b,c,['{0}: {1}'.format(k, v.item()) for k,v in loss_dict.items()]))
                    logger.info("res:{:.3f}".format(r))
                    a,b,c,d,e,f,g = 0,0,0,0,0,0,0

                # if iteration - start_iter > 5 and ((iteration + 1) % 50 == 0 or iteration == max_iter - 1):
                #     # print("epoch: ", epoch, "lr:", optimizer.param_groups[0]["lr"], "pse_acc:",pse_acc, "cluter_acc:",cluster_acc ,''.join(['{0}: {1}, '.format(k, v.item()) for k,v in loss_dict.items()]))
                #     r = a-b
                #     logger.info("epoch:{}, acc:{:.3f},pse_acc: {:.3f},base_acc:{:.3f},acc_new:{:.3f} ,loss :{:}".format(epoch,a,b,c,d,['{0}: {1}'.format(k, v.item()) for k,v in loss_dict.items()]))
                #     logger.info("res:{:.3f}".format(r))

                #     a,b,c,d,e,f,g = 0,0,0,0,0,0,0
                #     r1 += r

                # periodic_checkpointer.step(iteration)

                if iteration - start_iter > 5 and ((iteration + 1) % 200 == 0 or iteration == max_iter - 1):

                    model_student.eval()
                    print("Student model testing@", iteration)
                    test_sfda(cfg, model_student)

                    model_teacher.eval()
                    print("Teacher model testing@", iteration)
                    test_sfda(cfg, model_teacher)

                    logger.info("a1:{:.3f},b1:{:.3f}".format(a1,b1))
                    a1,b1 = 0,0

                # if iteration - start_iter > 5 and ((iteration + 1) % max_iter == 0 or iteration == max_iter - 1):

                #     new_student_dict = update_student_model(model_teacher, model_student, keep_rate=0.5)
                #     model_student.load_state_dict(new_student_dict)

            end_time = time.time()
            # training_time = end_time - start_time

            # print(f"Training Time for one epoch: {training_time:.4f} seconds")
            # print(f"Training Time for one step: {training_time/len_data_loader:.4f} seconds")

            # print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
            # print(f"Cached memory: {torch.cuda.memory_reserved()} bytes")

            # new_teacher_dict = update_teacher_model(model_student, model_teacher, keep_rate=0.9)
            # model_teacher.load_state_dict(new_teacher_dict)

            # logger.info("a1:{:.3f},b1:{:.3f},c1:{:.3f},a2:{:.3f},b2:{:.3f},c2:{:.3f}".format(a1,b1,c1,a2,b2,c2))


            # model_student.eval()
            # print("Student model testing@", epoch)
            # test_sfda(cfg, model_student)

            # model_teacher.eval()
            # print("Teacher model testing@", epoch)
            # test_sfda(cfg, model_teacher)
            
            # if epoch>=5:
            #     torch.save(model_teacher.state_dict(), cfg.OUTPUT_DIR + "/model_teacher_{}.pth".format(epoch))
            #     torch.save(model_student.state_dict(), cfg.OUTPUT_DIR + "/model_student_{}.pth".format(epoch))
        

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file('configs/sfda/sfda_foggy_mt.yaml')
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR+'/'+time_string
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    cfg = setup(args)
    model_student = build_model(cfg)
    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = "teacher_sfda_RCNN"
    cfg.freeze()
    model_teacher = build_model(cfg)
    model_teacher_s = build_model(cfg)

    logger.info("Model:\n{}".format(model_student))

    DetectionCheckpointer(model_student, save_dir=cfg.OUTPUT_DIR).load('source_model/cityscape_baseline/model_final.pth')
    DetectionCheckpointer(model_teacher, save_dir=cfg.OUTPUT_DIR).load('source_model/cityscape_baseline/model_final.pth')
    DetectionCheckpointer(model_teacher_s, save_dir=cfg.OUTPUT_DIR).load('source_model/cityscape_baseline/model_final.pth')

    logger.info("Trained model has been sucessfully loaded")
    return train_sfda(cfg, model_student, model_teacher,model_teacher_s)


if __name__ == "__main__":
    from datetime import datetime
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d-%H:%M:%S")

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
