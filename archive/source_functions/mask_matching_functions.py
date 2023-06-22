import os
import numpy as np
import skimage.io as io
import pandas as pd
from copy import deepcopy
import itertools
from source_functions.demo import resize_image, colour_pin_mask, find_top_label
from source_functions.main import find_aligned_label
from source_functions.alignment_helper_functions import adjust_alignment
from source_functions.label_merging import align_merged_label, new_corners


def return_image(id__, i, image_table, sample_pth, size_limit=2048):
    im = image_table[
        (image_table["id"] == id__) & (image_table["image_index"] == i + 1)
    ]["path"].iloc[0]
    try:
        img = io.imread(sample_pth + "/" + im)
    except:
        try:
            img = io.imread(sample_pth + "/" + im[:-4] + "JPG")
        except:
            img = io.imread(sample_pth + "/" + im[:-4] + "jpg")
    if np.shape(img)[1] > size_limit:
        img = resize_image(im)
    return img


def overlap_exclusion(msk_to_exclude_, msk_to_include_, tst):
    msk_to_exclude = tst[msk_to_exclude_]
    msk_to_include = tst[msk_to_include_]
    msk_to_exclude_edit = deepcopy(msk_to_exclude)
    msk_to_exclude_edit[np.where(msk_to_include == True)] = False
    return msk_to_exclude_edit


def review_overlaps(masks, filter_masks=True):

    n_ = np.shape(masks)[0]

    overlaps = []
    for i in range(n_):
        msk1 = masks[i]
        N1 = len(np.where(msk1 == True)[0])
        for j in range(i + 1, n_):
            msk2 = masks[j]
            N2 = len(np.where(msk2 == True)[0])
            N_overlap = len(np.where((msk1 == True) & (msk2 == True))[0])
            R1 = np.round(N_overlap / N1, 4)
            R2 = np.round(N_overlap / N2, 4)
            if (N_overlap != 0) and ((R1 > 0.15) or (R2 > 0.15)):
                overlaps.append([i, j, R1, R2, N_overlap])

    overlaps_sorted = deepcopy(overlaps)
    overlaps_sorted = sorted(overlaps_sorted, key=lambda x: x[4], reverse=True)

    masks_new = deepcopy(masks)

    for o in overlaps_sorted:
        msk1 = masks_new[o[0]]
        msk2 = masks_new[o[1]]
        N_overlap = len(np.where((msk1 == True) & (msk2 == True))[0])
        R1 = np.round(N_overlap / len(np.where(msk1 == True)[0]), 4)
        R2 = np.round(N_overlap / len(np.where(msk2 == True)[0]), 4)

        msk_to_exclude_ = [o[0], o[1]][np.argmax([R1, R2])]
        msk_to_include_ = [o[0], o[1]][np.argmin([R1, R2])]

        new_mask = overlap_exclusion(msk_to_exclude_, msk_to_include_, masks_new)

        masks_new[msk_to_exclude_] = new_mask

    for i in range(n_):
        msk_orig = masks[i]
        msk_new = masks_new[i]
        n1 = len(np.where(msk_orig == True)[0])
        n2 = len(np.where(msk_new == True)[0])
        r = n2 / n1
        if (filter_masks == True) and (r < 0.1):
            masks_new[i] = np.full(np.shape(masks_new[i]), False)

    return masks_new, len(overlaps)


def remove_small_masks(mask, limit=1000):
    masks_filtered = []
    for m in mask:
        p = len(np.where(m == True)[0])
        if p > limit:
            masks_filtered.append(m)
    return masks_filtered


def make_matches(midpoints, temp_midpoints):
    differences = {}
    matches_ind = {}
    for y in np.sort(temp_midpoints):
        v = abs(midpoints - y)
        differences[y] = v
        j = np.argmin(v)
        if j not in matches_ind.values():
            matches_ind[y] = j
        else:
            ky = [k for k in matches_ind.keys() if matches_ind[k] == j][0]
            diffs = differences[ky]
            p1 = diffs[j]
            p2 = v[j]
            if p1 <= p2:
                # original chosen match remains, but we choose the second closest match for current y
                j = np.argsort(v)[1]
                matches_ind[y] = j
            else:
                # change the original match, but keep the current match for the current y
                j_ = np.argsort(v)[1]
                matches_ind[y] = j
                matches_ind[ky] = j_

    return matches_ind


def make_matches2(matches_to_be_found, template_to_match):
    iterations = list(
        itertools.permutations(matches_to_be_found, len(template_to_match))
    )
    iterations_indices = list(
        itertools.permutations(enumerate(matches_to_be_found), len(template_to_match))
    )
    diffsum = [sum(abs(np.array(I) - np.sort(template_to_match))) for I in iterations]
    argmin = np.argmin(diffsum)
    matches = iterations[argmin]
    index = iterations_indices[argmin]
    matched_dict = {}
    for u, t in enumerate(np.sort(template_to_match)):
        matched_dict[t] = index[u][0]
    return matched_dict


def get_matched_masks_and_images(
    id__,
    image_table,
    all_info,
    mask_direc,
    sample_pth,
    max_no_labels=6,
    limit=1000,
    match_method=1,
):

    # 1) Get all masks for all images
    details_dict = {}
    msk_paths = image_table[image_table["id"] == id__]["mask_file"]

    for i, pth in enumerate(msk_paths):
        details_dict.update({i: {"mask_p": []}})
        bla = np.load(mask_direc + "/" + pth, allow_pickle=True)
        details_dict[i]["mask_p"] = bla

    # 2) Handle overlaps in masks
    all_images_ = []
    all_masks_ = []
    for q in range(0, 4):
        v = all_info[id__][q + 1]
        img_ = return_image(id__, q, image_table, sample_pth)
        tst = details_dict[v]["mask_p"]

        masks_new, _ = review_overlaps(tst)

        all_images_.append(img_)
        all_masks_.append(masks_new)

    # 3) Filter out small masks
    new_masks_ = [remove_small_masks(m, limit=limit) for m in all_masks_]

    # 4) Get number of masks to look at
    filtered_msk_count = [len(m) for m in new_masks_]
    min_count = min([min(filtered_msk_count), max_no_labels])

    # 5) Pick template
    k = np.argmin(filtered_msk_count)
    template_ = new_masks_[k]
    temp_vals = [len(np.where(m == True)[0]) for m in template_]
    template_sorted = np.array(template_)[np.argsort(temp_vals)[::-1][:min_count]]

    # 6) Find midpoint of masks in template
    yvals_temp = [
        sum([min(np.where(m == True)[0]), max(np.where(m == True)[0])]) / 2
        for m in template_sorted
    ]

    # 7) Compare midpoints of all masks to template
    final_label_masks = []
    for i in range(4):
        yv = [
            sum([min(np.where(m == True)[0]), max(np.where(m == True)[0])]) / 2
            for m in new_masks_[i]
        ]
        if match_method == 1:
            matches = make_matches(yv, yvals_temp)
        else:
            matches = make_matches2(yv, yvals_temp)

        msk_ = [new_masks_[i][p] for p in matches.values()]

        final_label_masks.append(msk_)

    return all_images_, final_label_masks
