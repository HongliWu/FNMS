import pickle as p
import numpy as np
import torch
import collections
from algorithms.metric import check_metric
from algorithms.nms_util import nms_whl, nms_whl_fusion
from ensemble_boxes import *


def nms_result(data, soft_nms=False, nms_thresh=0.05):
    gt_boxes = data['gt_boxes']
    gt_category = data['gt_category']
    boxes = data['boxes']
    scores = data['scores']
    category = data['category']

    batch_num = len(gt_category)
    batch_size = len(gt_category[0])
    oof = collections.defaultdict(list)
    for batch_id in range(batch_num):
        nms_bboxs, nms_scores = [], []
        for image_id in range(batch_size):
            boxes_image = boxes[batch_id][image_id]
            scores_image = scores[batch_id][image_id]
            boxes_image = torch.tensor(boxes_image).cuda()
            scores_image = torch.tensor(scores_image).cuda()
            if scores_image.sum() == 0:
                # no boxes to NMS, just return
                nms_scores.append(torch.zeros(0).cpu().numpy().copy())
                nms_bboxs.append(torch.zeros(0, 4).cpu().numpy().copy())
            else:
                nms_bbox, nms_score, _ = nms_whl(torch.cat([boxes_image, scores_image], dim=1),
                                                  nms_thresh,
                                                  soft_nms=soft_nms
                                                  )
                nms_bboxs.append(nms_bbox.cpu().numpy().copy())
                nms_scores.append(nms_score.cpu().numpy().copy())

        oof["boxes"].append(nms_bboxs)
        oof["scores"].append(nms_scores)
    oof["gt_boxes"] = gt_boxes
    oof["category"] = category
    oof["gt_category"] = gt_category
    return oof


def nms_fusion_result(data, soft_nms=False, nms_thresh=0.05, fusion_thresh=0.9):
    gt_boxes = data['gt_boxes']
    gt_category = data['gt_category']
    boxes = data['boxes']
    scores = data['scores']
    category = data['category']

    batch_num = len(gt_category)
    batch_size = len(gt_category[0])
    oof = collections.defaultdict(list)
    for batch_id in range(batch_num):
        nms_bboxs, nms_scores = [], []
        for image_id in range(batch_size):
            boxes_image = boxes[batch_id][image_id]
            scores_image = scores[batch_id][image_id]
            boxes_image = torch.tensor(boxes_image).cuda()
            scores_image = torch.tensor(scores_image).cuda()
            if scores_image.sum() == 0:
                # no boxes to NMS, just return
                nms_scores.append(torch.zeros(0).cpu().numpy().copy())
                nms_bboxs.append(torch.zeros(0, 4).cpu().numpy().copy())
            else:
                nms_bbox, nms_score, _ = nms_whl_fusion(torch.cat([boxes_image, scores_image], dim=1),
                                                        nms_thresh,
                                                        soft_nms=soft_nms,
                                                        fusion_thresh=fusion_thresh,
                                                        )
                nms_bboxs.append(nms_bbox.cpu().numpy().copy())
                nms_scores.append(nms_score.cpu().numpy().copy())

        oof["boxes"].append(nms_bboxs)
        oof["scores"].append(nms_scores)
    oof["gt_boxes"] = gt_boxes
    oof["category"] = category
    oof["gt_category"] = gt_category
    return oof


def nms_fusion_result_2(data_1, data_2, soft_nms=False, nms_thresh=0.05, fusion_thresh=0.9):
    gt_boxes_1 = data_1['gt_boxes']
    gt_category_1 = data_1['gt_category']
    boxes_1 = data_1['boxes']
    scores_1 = data_1['scores']
    category_1 = data_1['category']

    gt_boxes_2 = data_2['gt_boxes']
    gt_category_2 = data_2['gt_category']
    boxes_2 = data_2['boxes']
    scores_2 = data_2['scores']
    category_2 = data_2['category']

    batch_num = len(gt_category_1)
    batch_size = len(gt_category_1[0])
    oof = collections.defaultdict(list)
    for batch_id in range(batch_num):
        nms_bboxs, nms_scores = [], []
        for image_id in range(batch_size):

            assert np.all(gt_boxes_1[batch_id][image_id] == gt_boxes_2[batch_id][image_id]), "gt_boxes_1 and gt_boxes_2 are not same !"
            assert np.all(gt_category_1[batch_id][image_id] == gt_category_2[batch_id][image_id]), "gt_boxes_1 and gt_boxes_2 are not same !"

            boxes_image_1 = boxes_1[batch_id][image_id]
            scores_image_1 = scores_1[batch_id][image_id]
            boxes_image_2 = boxes_2[batch_id][image_id]
            scores_image_2 = scores_2[batch_id][image_id]

            if boxes_image_1.shape[0] == 0 and boxes_image_2.shape[0] == 0:
                boxes_image = boxes_image_1  # 都是空，随便等于一个就好
                scores_image = scores_image_1
            elif boxes_image_1.shape[0] != 0 and boxes_image_2.shape[0] == 0:
                boxes_image = boxes_image_1
                scores_image = scores_image_1
            elif boxes_image_1.shape[0] == 0 and boxes_image_2.shape[0] != 0:
                boxes_image = boxes_image_2
                scores_image = scores_image_2
            elif boxes_image_1.shape[0] != 0 and boxes_image_2.shape[0] != 0:
                boxes_image = np.vstack((boxes_image_1, boxes_image_2))
                scores_image = np.vstack((scores_image_1, scores_image_2))

            boxes_image = torch.tensor(boxes_image).cuda()
            scores_image = torch.tensor(scores_image).cuda()

            if scores_image.sum() == 0:
                # no boxes to NMS, just return
                nms_scores.append(torch.zeros(0).cpu().numpy().copy())
                nms_bboxs.append(torch.zeros(0, 4).cpu().numpy().copy())
            else:
                nms_bbox, nms_score, _ = nms_whl_fusion(torch.cat([boxes_image, scores_image], dim=1),
                                                        nms_thresh,
                                                        soft_nms=soft_nms,
                                                        fusion_thresh=fusion_thresh,
                                                        )
                nms_bboxs.append(nms_bbox.cpu().numpy().copy())
                nms_scores.append(nms_score.cpu().numpy().copy())

        oof["boxes"].append(nms_bboxs)
        oof["scores"].append(nms_scores)
    oof["gt_boxes"] = gt_boxes_1
    oof["category"] = category_1
    oof["gt_category"] = gt_category_1
    return oof


def soft_nms_result_2(data_1, data_2, nms_thresh=0.05):
    gt_boxes_1 = data_1['gt_boxes']
    gt_category_1 = data_1['gt_category']
    boxes_1 = data_1['boxes']
    scores_1 = data_1['scores']
    category_1 = data_1['category']

    gt_boxes_2 = data_2['gt_boxes']
    gt_category_2 = data_2['gt_category']
    boxes_2 = data_2['boxes']
    scores_2 = data_2['scores']
    category_2 = data_2['category']

    batch_num = len(gt_category_1)
    batch_size = len(gt_category_1[0])
    oof = collections.defaultdict(list)
    for batch_id in range(batch_num):
        nms_bboxs, nms_scores = [], []
        for image_id in range(batch_size):

            assert np.all(gt_boxes_1[batch_id][image_id] == gt_boxes_2[batch_id][image_id]), "gt_boxes_1 and gt_boxes_2 are not same !"
            assert np.all(gt_category_1[batch_id][image_id] == gt_category_2[batch_id][image_id]), "gt_boxes_1 and gt_boxes_2 are not same !"

            boxes_image_1 = boxes_1[batch_id][image_id]
            scores_image_1 = scores_1[batch_id][image_id]
            boxes_image_2 = boxes_2[batch_id][image_id]
            scores_image_2 = scores_2[batch_id][image_id]

            if boxes_image_1.shape[0] == 0 and boxes_image_2.shape[0] == 0:
                boxes_image = boxes_image_1  # 都是空，随便等于一个就好
                scores_image = scores_image_1
            elif boxes_image_1.shape[0] != 0 and boxes_image_2.shape[0] == 0:
                boxes_image = boxes_image_1
                scores_image = scores_image_1
            elif boxes_image_1.shape[0] == 0 and boxes_image_2.shape[0] != 0:
                boxes_image = boxes_image_2
                scores_image = scores_image_2
            elif boxes_image_1.shape[0] != 0 and boxes_image_2.shape[0] != 0:
                boxes_image = np.vstack((boxes_image_1, boxes_image_2))
                scores_image = np.hstack((scores_image_1, scores_image_2))

            #  start use weight boxes fusion
            boxes_image = np.expand_dims(boxes_image, 0) / 512
            scores_image = np.expand_dims(scores_image, 0)

            labels_image = scores_image * 0

            if scores_image.sum() == 0:
                # no boxes to NMS, just return
                nms_scores.append(torch.zeros(0).cpu().numpy().copy())
                nms_bboxs.append(torch.zeros(0, 4).cpu().numpy().copy())
            else:
                boxes_fusioned, scores_fusioned, labels_fusioned = soft_nms(boxes_image, scores_image, labels_image, iou_thr=nms_thresh, sigma=nms_thresh)
                nms_bboxs.append(boxes_fusioned * 512)
                nms_scores.append(scores_fusioned)

        oof["boxes"].append(nms_bboxs)
        oof["scores"].append(nms_scores)
    oof["gt_boxes"] = gt_boxes_1
    oof["category"] = category_1
    oof["gt_category"] = gt_category_1
    return oof


def nms_result_2(data_1, data_2, soft_nms=False, nms_thresh=0.05):
    gt_boxes_1 = data_1['gt_boxes']
    gt_category_1 = data_1['gt_category']
    boxes_1 = data_1['boxes']
    scores_1 = data_1['scores']
    category_1 = data_1['category']

    gt_boxes_2 = data_2['gt_boxes']
    gt_category_2 = data_2['gt_category']
    boxes_2 = data_2['boxes']
    scores_2 = data_2['scores']
    category_2 = data_2['category']

    batch_num = len(gt_category_1)
    batch_size = len(gt_category_1[0])
    oof = collections.defaultdict(list)
    for batch_id in range(batch_num):
        nms_bboxs, nms_scores = [], []
        for image_id in range(batch_size):

            assert np.all(gt_boxes_1[batch_id][image_id] == gt_boxes_2[batch_id][image_id]), "gt_boxes_1 and gt_boxes_2 are not same !"
            assert np.all(gt_category_1[batch_id][image_id] == gt_category_2[batch_id][image_id]), "gt_boxes_1 and gt_boxes_2 are not same !"

            boxes_image_1 = boxes_1[batch_id][image_id]
            scores_image_1 = scores_1[batch_id][image_id]
            boxes_image_2 = boxes_2[batch_id][image_id]
            scores_image_2 = scores_2[batch_id][image_id]

            if boxes_image_1.shape[0] == 0 and boxes_image_2.shape[0] == 0:
                boxes_image = boxes_image_1  # 都是空，随便等于一个就好
                scores_image = scores_image_1
            elif boxes_image_1.shape[0] != 0 and boxes_image_2.shape[0] == 0:
                boxes_image = boxes_image_1
                scores_image = scores_image_1
            elif boxes_image_1.shape[0] == 0 and boxes_image_2.shape[0] != 0:
                boxes_image = boxes_image_2
                scores_image = scores_image_2
            elif boxes_image_1.shape[0] != 0 and boxes_image_2.shape[0] != 0:
                boxes_image = np.vstack((boxes_image_1, boxes_image_2))
                scores_image = np.vstack((scores_image_1, scores_image_2))

            boxes_image = torch.tensor(boxes_image).cuda()
            scores_image = torch.tensor(scores_image).cuda()

            if scores_image.sum() == 0:
                # no boxes to NMS, just return
                nms_scores.append(torch.zeros(0).cpu().numpy().copy())
                nms_bboxs.append(torch.zeros(0, 4).cpu().numpy().copy())
            else:
                nms_bbox, nms_score, _ = nms_whl(torch.cat([boxes_image, scores_image], dim=1),
                                                 nms_thresh,
                                                 soft_nms=soft_nms
                                                 )
                nms_bboxs.append(nms_bbox.cpu().numpy().copy())
                nms_scores.append(nms_score.cpu().numpy().copy())

        oof["boxes"].append(nms_bboxs)
        oof["scores"].append(nms_scores)
    oof["gt_boxes"] = gt_boxes_1
    oof["category"] = category_1
    oof["gt_category"] = gt_category_1
    return oof


if __name__  == "__main__":
    before_nms_se_path = "../model_predicted_boxes/se_resnext101_dr0.75_512/store-nms-010.pkl"
    before_nms_res2net_path = "../model_predicted_boxes/res2net101_v1b_26w_4s_dr0.75_512/store-nms-012.pkl"
    after_nms_se_path = "../model_predicted_boxes/se_resnext101_dr0.75_512/valid_all_010.pkl"
    after_nms_res2net_path = "../model_predicted_boxes/res2net101_v1b_26w_4s_dr0.75_512/valid_all_012.pkl"

    before_nms_se = p.load(open(before_nms_se_path, 'rb'))
    before_nms_res2net = p.load(open(before_nms_res2net_path, 'rb'))
    after_nms_se = p.load(open(after_nms_se_path, 'rb'))
    after_nms_res2net = p.load(open(after_nms_res2net_path, 'rb'))

    # single model
    before_nms_se_soft_nmsed = nms_result(before_nms_se, soft_nms=True, nms_thresh=0.05)
    check_metric(before_nms_se_soft_nmsed, thresh_strage='cvprw')

    before_nms_res2net_soft_nmsed = nms_result(before_nms_res2net, soft_nms=True, nms_thresh=0.05)
    check_metric(before_nms_res2net_soft_nmsed, thresh_strage='cvprw')
    print('single model ↑')

    data_fusioned = soft_nms_result_2(after_nms_se, after_nms_res2net)
    check_metric(data_fusioned, thresh_strage='cvprw')
    print('model ensemble ↑')

