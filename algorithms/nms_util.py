import torch
import numpy as np


def iou_calc_whl(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def nms_whl(dets, thresh, variance=None, soft_nms=True):
    def nms_class(clsboxes):
        assert clsboxes.shape[1] == 5 or clsboxes.shape[1] == 9
        keep = []
        while clsboxes.shape[0] > 0:
            maxidx = torch.argmax(clsboxes[:, 4])
            maxbox = clsboxes[maxidx].unsqueeze(0)
            clsboxes = torch.cat((clsboxes[:maxidx], clsboxes[maxidx + 1:]), 0)
            # print("maxbox: ", maxbox)
            # print("clsboxes: ", clsboxes)
            if clsboxes.shape[0] == 0:
                keep.append(maxbox)
                continue
            else:
                iou = iou_calc_whl(maxbox[:, :4], clsboxes[:, :4])
            # KL VOTE
            if variance is not None:
                ioumask = iou > 0
                klbox = clsboxes[ioumask]
                klbox = torch.cat((klbox, maxbox), 0)
                kliou = iou[ioumask]
                klvar = klbox[:, -4:]
                pi = torch.exp(-1 * torch.pow((1 - kliou), 2) / 0.05)
                pi = torch.cat((pi, torch.ones(1).cuda()), 0).unsqueeze(1)
                pi = pi / klvar
                pi = pi / pi.sum(0)
                maxbox[0, :4] = (pi * klbox[:, :4]).sum(0)
            keep.append(maxbox)

            weight = torch.ones_like(iou)
            if not soft_nms:  # if not cfg.soft
                weight[iou > thresh] = 0  # weight[iou > cfg.nms_iou] = 0
            else:
                # weight = torch.exp(-1.0 * (iou ** 2 / cfg.softsigma))
                weight = torch.exp(-1.0 * (iou ** 2 / thresh))
            clsboxes[:, 4] = clsboxes[:, 4] * weight
            # filter_idx = (clsboxes[:, 4] >= cfg.score_thres).nonzero().squeeze(-1)
            filter_idx = (clsboxes[:, 4] >= 0.025).nonzero().squeeze(-1)
            clsboxes = clsboxes[filter_idx]
        return torch.cat(keep, 0).to(clsboxes.device)

    # bbox = boxes[:, :4].view(-1, 4)
    # numcls = boxes.shape[1] - 4
    # scores = boxes[:, 4:].view(-1, numcls)

    bbox = dets[:, :4].view(-1, 4)
    numcls = dets.shape[1] - 4  # cls == 1 in my experiment
    scores = dets[:, 4].view(-1, numcls)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []
    for i in range(numcls):
        # filter_idx = (scores[:, i] >= cfg.score_thres).nonzero().squeeze(-1)
        filter_idx = (scores[:, i] >= thresh).nonzero().squeeze(-1)
        if len(filter_idx) == 0:
            continue
        filter_boxes = bbox[filter_idx]
        filter_scores = scores[:, i][filter_idx].unsqueeze(1)
        if variance is not None:
            filter_variance = variance[filter_idx]
            clsbox = nms_class(torch.cat((filter_boxes, filter_scores, filter_variance), 1))
        else:
            clsbox = nms_class(torch.cat((filter_boxes, filter_scores), 1))
        if clsbox.shape[0] > 0:
            picked_boxes.append(clsbox[:, :4])
            picked_score.append(clsbox[:, 4])
            picked_label.extend([torch.ByteTensor([i]) for _ in range(len(clsbox))])
    if len(picked_boxes) == 0:
        return torch.tensor(picked_boxes), torch.tensor(picked_score), torch.tensor(picked_label)
        # return None, None, None
    else:
        return torch.cat(picked_boxes), torch.cat(picked_score), torch.cat(picked_label)


def nms_whl_fusion(dets, thresh, variance=None, soft_nms=True, fusion_thresh=0.9):
    def nms_class(clsboxes):
        assert clsboxes.shape[1] == 5 or clsboxes.shape[1] == 9
        keep = []
        while clsboxes.shape[0] > 0:
            maxidx = torch.argmax(clsboxes[:, 4])
            maxbox = clsboxes[maxidx].unsqueeze(0)
            clsboxes = torch.cat((clsboxes[:maxidx], clsboxes[maxidx + 1:]), 0)
            if clsboxes.shape[0] == 0:
                keep.append(maxbox)
                continue
            else:
                iou = iou_calc_whl(maxbox[:, :4], clsboxes[:, :4])

            # add fusion
            mask_fusion = iou > fusion_thresh
            if mask_fusion.sum() > 0:
                need_fusinon_box = torch.cat((maxbox, clsboxes[mask_fusion]), dim=0)
                scores_sum = need_fusinon_box[:, 4].sum()
                need_fusinon_weights = (need_fusinon_box[:, 4] / scores_sum).repeat(5, 1).t()
                weighted_need_fusinon_box = need_fusinon_box * need_fusinon_weights
                fusion_box = weighted_need_fusinon_box.sum(dim=0).unsqueeze(0)
                keep.append(fusion_box)
            else:
                keep.append(maxbox)

            weight = torch.ones_like(iou)
            if not soft_nms:  # if not cfg.soft
                weight[iou > thresh] = 0  # weight[iou > cfg.nms_iou] = 0
            else:
                # weight = torch.exp(-1.0 * (iou ** 2 / cfg.softsigma))
                weight = torch.exp(-1.0 * (iou ** 2 / thresh))
            clsboxes[:, 4] = clsboxes[:, 4] * weight
            # filter_idx = (clsboxes[:, 4] >= cfg.score_thres).nonzero().squeeze(-1)
            filter_idx = (clsboxes[:, 4] >= 0.025).nonzero().squeeze(-1)
            clsboxes = clsboxes[filter_idx]
        return torch.cat(keep, 0).to(clsboxes.device)

    # bbox = boxes[:, :4].view(-1, 4)
    # numcls = boxes.shape[1] - 4
    # scores = boxes[:, 4:].view(-1, numcls)

    bbox = dets[:, :4].view(-1, 4)
    numcls = dets.shape[1] - 4  # cls == 1 in my experiment
    scores = dets[:, 4].view(-1, numcls)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []
    for i in range(numcls):
        # filter_idx = (scores[:, i] >= cfg.score_thres).nonzero().squeeze(-1)
        filter_idx = (scores[:, i] >= thresh).nonzero().squeeze(-1)
        if len(filter_idx) == 0:
            continue
        filter_boxes = bbox[filter_idx]
        filter_scores = scores[:, i][filter_idx].unsqueeze(1)
        if variance is not None:
            filter_variance = variance[filter_idx]
            clsbox = nms_class(torch.cat((filter_boxes, filter_scores, filter_variance), 1))
        else:
            clsbox = nms_class(torch.cat((filter_boxes, filter_scores), 1))
        if clsbox.shape[0] > 0:
            picked_boxes.append(clsbox[:, :4])
            picked_score.append(clsbox[:, 4])
            picked_label.extend([torch.ByteTensor([i]) for _ in range(len(clsbox))])
    if len(picked_boxes) == 0:
        return torch.tensor(picked_boxes), torch.tensor(picked_score), torch.tensor(picked_label)
        # return None, None, None
    else:
        return torch.cat(picked_boxes), torch.cat(picked_score), torch.cat(picked_label)

