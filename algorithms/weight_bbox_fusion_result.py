import pickle as p
import numpy as np
import torch
import collections
from ensemble_boxes import *
from algorithms.metric import check_metric


def wbf_result(data_1, data_2, soft_nms=False, nms_thresh=0.05, fusion_thresh=0.9):
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
                boxes_fusioned, scores_fusioned, labels_fusioned = weighted_boxes_fusion(boxes_image,
                                                                                         scores_image,
                                                                                         labels_image,
                                                                                         )
                nms_bboxs.append(boxes_fusioned * 512)
                nms_scores.append(scores_fusioned)

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

    after_nms_se = p.load(open(after_nms_se_path, 'rb'))
    after_nms_res2net = p.load(open(after_nms_res2net_path, 'rb'))

    # model ensemble
    data_fusioned = wbf_result(after_nms_se, after_nms_res2net, nms_thresh=0.05, fusion_thresh=1.)
    check_metric(data_fusioned, thresh_strage='cvprw')
    print('model ensemble ↑')