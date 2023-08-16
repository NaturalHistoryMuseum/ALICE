"""
ALICE SOFTWARE V2
-----------------

We list out the steps that make up the new ALICE V2 software. 
We note that all main CNN related functions are based on the Detectron software: https://github.com/facebookresearch/detectron2

STEPS:
1) Label segmentation on all four images. (Uses trained detectron model.)
2) Label mask grouping, i.e., index and match each mask across the four images.
--- these next steps loop through each group of masks across the images ---
3) Box approximating for each of the 4 masks.
4) Warp the label box so that it's in a standard 2D viewpoint.
5) Minor alignments and rotations on each of the 4 warped labels. (Rotation options 0,90,180,270.)
6) Select template amongst the 4 warped and adjusted labels.
7) Align labels to the template.
8) Remove outliers from list of aligned labels (e.g., badly aligned images). (Uses trained detectron model.)
9) Merge labels into one. 
10) Classify merged label as good or bad.

For one function that includes all the following steps, refer to the file ALICE_main_function.py.

"""

###########################################################################
###########################################################################

###########
# IMPORTS #
###########

"""
----------------------------
IMPORTS FROM STEP 1 / STEP 8
----------------------------
"""

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import os

"""
--------------------
IMPORTS FROM STEP 2
--------------------
"""

from mask_matching_functions import review_overlaps, remove_small_masks, make_matches_v2
import numpy as np

"""
--------------------
IMPORTS FROM STEP 3
--------------------
"""

from label_transformation import (
    find_label_corners,
    backup_corner_method,
    define_label_sides,
    combine_masks,
    compare_corners,
    reconfigure_corner_global,
    check_corners,
)
from skimage import measure

"""
--------------------
IMPORTS FROM STEP 4
--------------------
"""

from label_transformation import perspective_transform

"""
--------------------
IMPORTS FROM STEP 5
--------------------
"""

from alignment_helper_functions import adjust_alignment
import imutils
from copy import deepcopy
from label_merging import (
    max_ocr_length_orientation,
    tesseract_orientation_with_binarization_thresholding,
)

"""
--------------------
IMPORTS FROM STEP 6
--------------------
"""

import pytesseract
import re

"""
--------------------
IMPORTS FROM STEP 7
--------------------
"""

from label_merging import align_warped_label_to_template

"""
--------------------
IMPORTS FROM STEP 9
--------------------
"""

from label_merging import merge_label


###########################################################################
###########################################################################

##############
# FINAL CODE #
##############


"""
----------------------
STEP 1 -- SEGMENTATION
----------------------
"""

####################
# Variables to edit:
####################
SEGMENTATION_THRESHOLD_VALUE = 0.7
PATH_TO_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
PATH_TO_WEIGHTS = "/content/drive/My Drive/ALICE/model_final.pth"

###########################################################################

setup_logger()

# Model setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(PATH_TO_MODEL))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# Predictor
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, PATH_TO_WEIGHTS)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SEGMENTATION_THRESHOLD_VALUE
segmentation_predictor = DefaultPredictor(cfg)

###########################################################################

############
# Functions
############


def segment_labels_from_image_path(image):
    # Input: image
    # Output: masks (np.array)
    outputs = segmentation_predictor(image)
    label_masks = outputs["instances"].to("cpu").pred_masks.numpy()
    return label_masks, image


"""
-----------------------------
STEP 2 -- LABEL MASK GROUPING
-----------------------------
"""

####################
# Variables to edit:
####################
MAX_NUMBER_OF_LABELS = 6  # Maximum number of labels to look at per image.

###########################################################################

############
# Functions
############


def match_labels_across_images(all_masks, max_no_labels=MAX_NUMBER_OF_LABELS):
    # Input: all masks across all 4 images. (np.array)
    # Output: filtered masks across all 4 images. (np.array)

    # 1) Filter masks and remove overlapping regions:
    all_masks_edited = []  # set of masks after potential overlaps were removed.
    for i in range(4):
        masks = all_masks[i][0]
        masks_new, _ = review_overlaps(masks)  # remove overlaps between masks
        masks_new = remove_small_masks(
            masks_new, limit=1500
        )  # exclude masks that are too small.
        all_masks_edited.append(masks_new)

    # 2) Define number of labels to look at:
    # Note that this number will be less than the maximum number of labels defined with the variable max_no_labels.
    filtered_msk_count = [
        len(m) for m in all_masks_edited
    ]  # Count of filtered masks per image.
    min_count = min(
        [min(filtered_msk_count), max_no_labels]
    )  # Select number based on image with the fewest remaining labels.

    # 3) Select template image/mask:
    # The masks will be matched across images, based on a template.
    k = np.argmin(filtered_msk_count)
    template = all_masks_edited[k]
    template_mask_sizes = [len(np.where(m == True)[0]) for m in template]
    template_sorted = np.array(template)[
        np.argsort(template_mask_sizes)[::-1][:min_count]
    ]  # sort template based on mask size.

    # 4) Find midpoints of masks in template:
    template_mask_midpoints_y = [
        sum([min(np.where(m == True)[0]), max(np.where(m == True)[0])]) / 2
        for m in template_sorted
    ]

    # 5) Compare midpoints of all masks to template:
    all_final_masks = []
    for i in range(4):
        mask_midpoints_y = [
            sum([min(np.where(m == True)[0]), max(np.where(m == True)[0])]) / 2
            for m in all_masks_edited[i]
        ]
        label_matches_index = make_matches_v2(
            mask_midpoints_y, template_mask_midpoints_y
        )  # match labels based on the y coordinate of the midpoints.
        matched_masks = [all_masks_edited[i][p] for p in label_matches_index.values()]
        all_final_masks.append(matched_masks)

    return all_final_masks


"""
---------------------------
STEP 3 -- BOX APPROXIMATING
---------------------------
"""

####################
# Variables to edit:
####################
CORNER_FINDING_METHOD = 0
# Standard method of corner finding, else backup method will be used.

###########################################################################

############
# Functions
############


def define_box_around_label(
    label_mask,
    image,
    corner_finding_method=CORNER_FINDING_METHOD,
    combine_label_masks=True,
):
    # Input: label masks and images.
    # Output: corners of label, distances between corners.

    # 1) Find contour around mask:
    contours_around_mask = measure.find_contours(label_mask, 0.8)
    # 2) Check how many contours are found in mask:
    # If multiple, then we can combine masks:
    if (len(contours_around_mask) > 1) and (combine_label_masks is True):
        label_mask = combine_masks(label_mask, contours_around_mask)
        contours_around_mask = measure.find_contours(label_mask, 0.8)
    # 3) Find corners of label:
    if corner_finding_method == 0:
        corners_x, corners_y = find_label_corners(label_mask, contours_around_mask)
    else:
        contours_reshaped = [
            contours_around_mask[0][:, 1],
            contours_around_mask[0][:, 0],
        ]
        corners_x, corners_y = backup_corner_method(contours_reshaped)
    # 4) Define the long and short sides of label:
    short_side_index, long_side_index, _, distance_between_corners = define_label_sides(
        corners_x, corners_y
    )

    # 5) Check corners [used only if backup corner finding method not originally used]:
    if corner_finding_method == 0:
        cc = check_corners(corners_x, corners_y, short_side_index, long_side_index)
        # If corners don't fit the criteria, redo the corner computation:
        if cc == True:
            try:
                contours_reshaped = [
                    contours_around_mask[0][:, 1],
                    contours_around_mask[0][:, 0],
                ]
                corners_x, corners_y = backup_corner_method(contours_reshaped)
                (
                    short_side_index,
                    long_side_index,
                    _,
                    distance_between_corners,
                ) = define_label_sides(corners_x, corners_y)
            except:
                pass

    # 6) Edit corners:
    corners_x_updated, corners_y_updated = reconfigure_corner_global(
        short_side_index,
        long_side_index,
        distance_between_corners,
        corners_x,
        corners_y,
        method="both",
        original_mask=label_mask,
        original_image=image,
    )
    # 6) Check corners:
    # If these new corners are significantly larger than the original corners (before last edit), we
    # use the original corners instead. Check this with compare_corners function:
    corners_x_updated, corners_y_updated = compare_corners(
        image, corners_x, corners_y, corners_x_updated, corners_y_updated
    )

    return (
        corners_x_updated,
        corners_y_updated,
        distance_between_corners,
        short_side_index,
        long_side_index,
    )


"""
---------------------
STEP 4 -- WARP LABELS
---------------------
"""

####################
# Variables to edit:
####################
MINIMUM_LABEL_SIZE = 20  # minimum size of length or width in pixels.

###########################################################################

############
# Functions
############


def warp_label(
    corner_results, image, original_mask, minimum_label_size=MINIMUM_LABEL_SIZE
):
    # Input: Corners of labels (results from define_box_around_label), original image / mask.
    # Output: Four warped labels (after perspective transformation).

    (
        corners_x_updated,
        corners_y_updated,
        distance_between_corners,
        short_side_index,
        long_side_index,
    ) = corner_results

    warped_label, _ = perspective_transform(
        corners_x_updated,
        corners_y_updated,
        image,
        distance_between_corners,
        short_side_index,
        long_side_index,
    )

    if any(np.array(np.shape(warped_label)[:2]) < minimum_label_size):
        corner_results = define_box_around_label(
            original_mask, image, corner_finding_method=1, combine_label_masks=False
        )
        (
            corners_x_updated,
            corners_y_updated,
            distance_between_corners,
            short_side_index,
            long_side_index,
        ) = corner_results
        warped_label, _ = perspective_transform(
            corners_x_updated,
            corners_y_updated,
            image,
            distance_between_corners,
            short_side_index,
            long_side_index,
        )

    return warped_label


"""
---------------------------------------
STEP 5 -- MINOR ADJUSTMENTS & ROTATIONS
---------------------------------------
"""

############
# Functions
############


def adjust_and_rotate_warped_label(warped_label):
    # Input: warped label.
    # Output: Optimally rotated label.

    # 1) Minor adjustment in label angle:
    optimal_angle = adjust_alignment(warped_label, np.linspace(-10, 10, 21))[0]
    adjusted_label = imutils.rotate_bound(warped_label, optimal_angle)
    # 2) Check the orientation of label with an optimized tesseract orientation tool:
    # Note that as this method isn't up to standard yet, we only accept the orientation
    # based on additional criteria, which also includes a backup tool to predict orientation.
    # This backup tool simply rotates the images 0,90,180,270 degrees and counts the number of
    # letters in each binarized image, using tesseract.
    (
        orientation,
        angle_to_rotate,
        confidence,
        best_binarized_image,
    ) = tesseract_orientation_with_binarization_thresholding(adjusted_label)
    if (orientation == 180) or (confidence > 0.5):
        rotated_label = imutils.rotate_bound(adjusted_label, angle_to_rotate)
    else:
        new_orientation_angle = max_ocr_length_orientation(adjusted_label)
        if new_orientation_angle == angle_to_rotate:
            rotated_label = imutils.rotate_bound(adjusted_label, angle_to_rotate)
        else:
            rotated_label = deepcopy(adjusted_label)
    return rotated_label, best_binarized_image


"""
--------------------------------------
STEP 6 -- SELECT TEMPLATE WARPED LABEL
--------------------------------------
"""

############
# Functions
############


def select_template_warped_label(all_rotated_warped_labels, all_binarized_labels):
    # Input: rotated_labels, binarized labels
    # Output: template based on one of the four rotated_labels.

    longest_label_text_length = 0
    template_warped_label = all_rotated_warped_labels[0]
    for ind, rotated_label in enumerate(all_rotated_warped_labels):
        binarized_label = all_binarized_labels[ind]
        ocr_results = pytesseract.image_to_string(
            binarized_label, config="--psm 11 script=Latin"
        )
        text_in_label = " ".join(re.findall("\w+", ocr_results))
        if len(text_in_label) > longest_label_text_length:
            longest_label_text_length = len(text_in_label)
            template_warped_label = deepcopy(rotated_label)
    return template_warped_label


"""
----------------------------------
STEP 7 -- ALIGN LABELS TO TEMPLATE
----------------------------------
"""

############
# Functions
############


def align_warped_labels(all_rotated_warped_labels, template_warped_label):
    # Input: all rotated labels, template_label
    # Ouput: all labels aligned to template.
    aligned_labels = []
    for warped_label in all_rotated_warped_labels:
        if warped_label == template_warped_label:
            aligned_labels.append(warped_label)
        else:
            try:
                aligned_label = align_warped_label_to_template(
                    warped_label, template_warped_label
                )[0]
                aligned_labels.append(aligned_label)
            except:
                pass

    if len(aligned_labels) == 0:
        aligned_labels = deepcopy(all_rotated_warped_labels)
    return aligned_labels


"""
-------------------------------------
STEP 8 -- DETECT & EXCLUDE BAD LABELS
-------------------------------------
"""

####################
# Variables to edit:
####################
CLASSIFICATION_THRESHOLD_VALUE = 0.7
PATH_TO_BAD_LABEL_DETECTION_WEIGHTS = (
    "/content/drive/My Drive/ALICE/model_bad_label_final.pth"
)

###########################################################################

# Model setup
del cfg
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(PATH_TO_MODEL))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

# Predictor
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, PATH_TO_BAD_LABEL_DETECTION_WEIGHTS)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CLASSIFICATION_THRESHOLD_VALUE
good_or_bad_label_predictor = DefaultPredictor(cfg)

###########################################################################

############
# Functions
############


def is_label_good(label):
    # Input: label
    # Output: classification of whether label is good or bad (np.bool).
    classification = good_or_bad_label_predictor(label)
    good_or_bad = classification["instances"].pred_classes[0].item()
    return np.bool(good_or_bad)


def exclude_bad_labels_with_template_pixel_difference(
    aligned_labels, template_warped_label
):
    # Input: aligned labels, template label.
    # Output: filtered labels (that are not too different from template).
    sum_of_pixel_differences_all = [
        np.sum(abs(template_warped_label - label)) for label in aligned_labels
    ]
    sum_of_pixel_differences = [
        diffsum for diffsum in sum_of_pixel_differences_all if diffsum > 0
    ]
    upper_bound_pixel_diffsum = np.median(sum_of_pixel_differences) + (
        np.percentile(sum_of_pixel_differences, 75)
        - np.percentile(sum_of_pixel_differences, 25)
    )
    filtered_aligned_labels = []
    for ind, label in enumerate(aligned_labels):
        if sum_of_pixel_differences_all[ind] < upper_bound_pixel_diffsum:
            filtered_aligned_labels.append(label)
    return filtered_aligned_labels


def find_all_good_labels(aligned_labels, template_warped_label):
    # Input: aligned labels
    # Output: good labels.
    try:
        filtered_aligned_labels = exclude_bad_labels_with_template_pixel_difference(
            aligned_labels, template_warped_label
        )
    except:
        filtered_aligned_labels = deepcopy(aligned_labels)
    filtered_labels = []
    for label in filtered_aligned_labels:
        if is_label_good(label) is True:
            filtered_labels.append(label)
    if len(filtered_labels) == 0:
        filtered_labels = deepcopy(aligned_labels)
    return filtered_labels


"""
----------------------
STEP 9 -- MERGE LABELS
----------------------
"""

############
# Functions
############


def merge_aligned_labels(filtered_labels):
    # Input: all filtered labels.
    # Output: merged label.
    try:
        merged_label = merge_label(filtered_labels, method=0)
    except:
        merged_label = merge_label(filtered_labels, method=1)
    return merged_label


###########################################################################
###########################################################################

##############
# FINAL CODE #
##############
