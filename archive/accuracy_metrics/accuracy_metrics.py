import numpy as np
import skimage.draw
import pandas as pd
from sklearn.metrics import auc

# Mask functions
#################


def create_info(annotations, height, width):
    """Reformat annotations into dictionary format along with height and width
    information. This is will then be fed into the extract_bboxes function.
    Input: ground truth annotations, height and width of image.
    Output: annotation in dictionary format.
    """
    info = {}
    info["height"] = height
    info["width"] = width
    n = len(annotations)
    polygons = []
    for i in range(0, n):
        polygons.append(annotations[i]["shape_attributes"])

    info["polygons"] = polygons
    return info


def ground_truth_masks(data, filename, height=1365, width=2048):
    """Creates masks from ground truth annotations.
    Input: metadata (from json) containing all annotations, filename,
    height of image, and width of image.
    Output: masks.
    """

    info = create_info(data[filename]["regions"], height, width)
    mask = np.zeros(
        [info["height"], info["width"], len(info["polygons"])], dtype=np.uint8
    )

    for i, p in enumerate(info["polygons"]):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
        mask[rr, cc, i] = 1

    return mask


def reshape_mask(mask):
    """Used to reshape masks resulting from detectron model."""
    reshaped = np.zeros((np.shape(mask)[1], np.shape(mask)[2], np.shape(mask)[0]))
    for i, I in enumerate(mask):
        reshaped[:, :, i] = I
    return reshaped


# Bounding box functions
########################


def extract_bboxes(mask):
    """Computes bounding boxes from masks.
    Input: mask, [height, width, num_instances]. Mask pixels are either 1 or 0.
    Output: bounding box array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def match_boxes(boxes_new, boxes_old):
    """Match bounding boxes from predictions to those from ground truths.
    Input: bounding boxes
    Output: index of bounding boxes from ground truth for each predicted bounding box.
    """
    pairs_matched = []
    for i, k in enumerate(boxes_new):
        min_sum = 1e5
        min_ind = 0
        for j, l in enumerate(boxes_old):
            v = sum(abs(k - l))
            if v < min_sum:
                min_sum = v
                min_ind = j
        pairs_matched.append([i, min_ind])

    return pairs_matched


# IOU functions
################


def cal_iou(mask1, mask2):
    """Computes IOU between individual masks.
    Input: two masks
    Output: one IOU value.
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.sum(mask1 * mask2)
    union = np.sum((mask1 + mask2).astype(bool))
    return intersection / union


def compute_iou(b1, b2):
    """Computes IOU between individual boxes.
    Input: two individual boxes.
    Output: one IoU value.
    """
    # 1. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = b1
    b2_y1, b2_x1, b2_y2, b2_x2 = b2
    y1 = max(b1_y1, b2_y1)
    x1 = max(b1_x1, b2_x1)
    y2 = min(b1_y2, b2_y2)
    x2 = min(b1_x2, b2_x2)
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    # 2. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 3. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    return iou


def seg_box_iou(M1, M2, masktype="box"):
    """Function to calculate IOU between boxes or masks."""
    if masktype == "box":
        iou = compute_iou(M1, M2)
    else:
        iou = cal_iou(M1, M2)
    return iou


# Global accuracy metric functions
###################################


def compute_TP_FP_dict(
    boxes_new, boxes_old, pairs_matched, image_name, mtype="box", scores=None, k=0.5
):
    """Classify detections as either true positive (TP) or false positive (FP)
    based on iou between mask or bounding box of prediction and ground truth.
    Input: masks or bounding boxes, index of masks / bounding box matches (see func match_boxes),
    confidence scores, threshold value (k) for iou comparisons.
    Output: dictionary containing TP/FP details for each detection.
    """

    new_order = np.argsort(
        [
            seg_box_iou(boxes_new[ij[0]], boxes_old[ij[1]], masktype=mtype)
            for ij in pairs_matched
        ]
    )
    pairs_iou_ordered = np.array(pairs_matched)[new_order][::-1]

    if scores is None:
        scores = np.zeros(len(boxes_new))

    dict_ = {}
    matches = []

    for ij in pairs_iou_ordered:
        i, j = ij
        iou = np.round(seg_box_iou(boxes_new[i], boxes_old[j], masktype=mtype), 5)
        score = np.round(scores[i], 5)
        if iou >= k:
            if j not in matches:
                matches.append(j)
                dict_[i] = {
                    "id": image_name + "_" + str(i),
                    "TP": 1,
                    "FP": 0,
                    "score": score,
                    "iou": iou,
                }
            else:
                dict_[i] = {
                    "id": image_name + "_" + str(i),
                    "TP": 0,
                    "FP": 1,
                    "score": score,
                    "iou": iou,
                }
        else:
            dict_[i] = {
                "id": image_name + "_" + str(i),
                "TP": 0,
                "FP": 1,
                "score": score,
                "iou": iou,
            }

    return dict_


def get_TP_FP(
    pred_path, filename, image_name, data, iou_thresholds=[0.5], maskType="box"
):
    """Input: filename, path to predicted masks for specific image,
    metadata containing all ground truth annotations, iou thresholds.
    Output: list of dictionaries containing TP and FP classifications for
    each detection, for each given iou threshold. Function also outputs
    total number of predictions and ground truths.
    """
    # 1) Load predictions for image
    results = np.load(pred_path, allow_pickle=True)
    try:
        pred_masks = results.item()["masks"]
    except:
        pred_masks = reshape_mask(results)

    # 2) Get confidence scores
    try:
        scores = results.item()["scores"]
    except:
        scores = np.zeros(len(results))

    # 2) Get shape of image
    try:
        a, b = np.shape(results.item()["masks"])[:2]
    except:
        a, b = np.shape(results)[1:]

    # 3) Load ground truths into masks
    mask = ground_truth_masks(data, filename, a, b)

    # 4) Find bounding boxes for the ground truth and predictions
    boxes_old = extract_bboxes(mask)
    boxes_new = extract_bboxes(pred_masks)

    tot_pred = len(boxes_new)
    tot_grnd = len(boxes_old)

    # 5) Match predictions to ground truth boxes
    pairs_matched = match_boxes(boxes_new, boxes_old)

    # 6) Compute TPs and FPs per threshold
    B1 = boxes_new
    B2 = boxes_old
    if maskType == "mask":
        B1 = []
        for ii in range(0, np.shape(pred_masks)[-1]):
            B1.append(pred_masks[:, :, ii])
        B2 = []
        for ii in range(0, np.shape(mask)[-1]):
            B2.append(mask[:, :, ii])
    dfs = [
        compute_TP_FP_dict(
            B1,
            B2,
            pairs_matched,
            image_name,
            mtype=maskType,
            scores=scores,
            k=T,
        )
        for T in iou_thresholds
    ]

    return dfs, tot_pred, tot_grnd


def get_pr_curve_and_AP(df, total_ground, sorting="score"):
    """Compute precision and recall and the Average Precision score (AP).
    Note that usually we will sort by confidence score (bb). Alternative could
    # be to sort by iou.
    Input: dataframe containing TP and FP classifications for each detection,
    as well as confidence scores (used for sorting) and total number of ground truths.
    Output: updated dataframe containing precision and recall columns, and AP score.
    """
    # 1) Sort by confidence score:
    df_sorted = df.sort_values(by=[sorting], ascending=False).reset_index(drop=True)

    # 2) Accumlated TPs and FPs:
    df_sorted["TP_acc"] = np.cumsum(df_sorted["TP"])
    df_sorted["FP_acc"] = np.cumsum(df_sorted["FP"])

    # 3) Compute precision / recall per detection:
    df_sorted["precision"] = df_sorted["TP_acc"] / (
        df_sorted["TP_acc"] + df_sorted["FP_acc"]
    )
    df_sorted["recall"] = df_sorted["TP_acc"] / total_ground

    # 4) Compute AP (by area under PR curve):
    AP = auc(df_sorted["recall"], df_sorted["precision"])

    return df_sorted, AP
