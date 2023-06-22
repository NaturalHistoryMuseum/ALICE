# Imports #
###########

import os
import sys
import random
import numpy as np
import skimage.io
import tqdm
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")
import config

# Import Mask RCNN
import utils
import model as modellib
import visualize

# Import COCO config
import coco
import re
import pandas as pd
from main import find_aligned_label
from label_merging import align_merged_label
from alignment_helper_functions import adjust_alignment
import PIL
import skimage.io as io
import matplotlib.pyplot as plt


# Basic functions #
###################


def resize_image(image_path, basewidth=2048, path=True):
    if path == True:
        image = PIL.Image.open(image_path)
    else:
        image = deepcopy(image_path)
    wpercent = basewidth / float(image.size[0])
    hsize = int((float(image.size[1]) * float(wpercent)))
    img = image.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    I = np.asarray(img)

    return I


def colour_pin_mask(masks, image, new_col=[255, 0, 0]):
    msk_img = deepcopy(image)
    v = np.shape(masks)[-1]
    for i in range(0, v):
        msk_img[np.where(masks[:, :, i] == True)] = new_col
    return msk_img


def find_top_label(msks_, method="highest", bound_prct=25):

    all_lens = []
    p = np.shape(msks_)[-1]
    for i in range(0, p):
        all_lens.append(len(np.where(msks_[:, :, i] == True)[0]))

    if method == "highest":

        try:
            bound = np.percentile(all_lens, bound_prct)

            a = [
                min(np.where(msks_[:, :, i] == True)[0])
                if len(np.where(msks_[:, :, i] == True)[0]) > 0
                else 1e9
                for i in range(0, p)
                if all_lens[i] > bound
            ]

            b = [i for i in range(0, p) if all_lens[i] > bound]

            k = b[np.argmin(a)]
        except:
            a = [
                min(np.where(msks_[:, :, i] == True)[0])
                if len(np.where(msks_[:, :, i] == True)[0]) > 0
                else 1e9
                for i in range(0, p)
            ]
            k = np.argmax(a)

    elif method == "largest":
        k = np.argmax(all_lens)
    elif method == "lowest":
        a = [
            max(np.where(msks_[:, :, i] == True)[0])
            if len(np.where(msks_[:, :, i] == True)[0]) > 0
            else 0
            for i in range(0, p)
        ]
        k = np.argmax(a)

    return msks_[:, :, k]


# CNN functions #
#################


def load_label_model(weights_path="drive/My Drive/ALICE/mask_rcnn_label_resized.h5"):
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 2

    config_ = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=" ", config=config_)

    # Load weights trained on ALICE data
    model.load_weights(weights_path, by_name=True)

    return model


def cnn_segment(
    class_names,
    image_path,
    plot=True,
    loaded_image=[],
    weights_path="drive/My Drive/ALICE/mask_rcnn_label_resized.h5",
    weights=False,
    model=None,
):
    # Input the weights as "model" if it has been loaded already and set weights=True. Else, set weights=False.
    if weights == False:
        model = load_label_model(weights_path)

    # Load image
    if len(loaded_image) == 0:
        image = skimage.io.imread(image_path)
    else:
        image = deepcopy(loaded_image)

    # Run detection
    results = model.detect([image], verbose=0)
    r = results[0]

    if plot == True:
        fig, ax = plt.subplots()
        # Visualize results
        visualize.display_instances(
            image,
            r["rois"],
            r["masks"],
            r["class_ids"],
            class_names,
            r["scores"],
            ax=ax,
        )

    res = r["masks"]
    return image, res


# General Functions #
#####################

# Paths
path_w_lab = "drive/My Drive/ALICE/mask_rcnn_label_resized.h5"
path_w_pin = "drive/My Drive/ALICE/mask_rcnn_pin.h5"
classes_lab = ["BG", "label"]
classes_pin = ["BG", "pin"]


def get_masks_(
    paths,
    with_pin=False,
    multiple_images=True,
    filter_mask="highest",
    pin_path=path_w_pin,
    label_path=path_w_lab,
    model=None,
):

    if model == None:
        model = load_label_model(label_path)

    details_dict = {}
    for i, path_i in enumerate(paths):
        img = io.imread(path_i)
        if np.shape(img)[1] > 2048:
            img = resize_image(path_i)

        _, mask_label = cnn_segment(
            classes_lab,
            path_i,
            plot=False,
            loaded_image=img,
            weights=True,
            weights_path=path_w_lab,
            model=model,
        )

        if filter_mask != None:
            label = find_top_label(mask_label, method=filter_mask)
            if multiple_images == True:
                details_dict.update({i: {"image": img, "mask_l": label, "mask_p": []}})

    if with_pin == True:
        model = load_label_model(pin_path)
        for i, path_i in enumerate(paths):
            img = details_dict[i]["image"]
            _, mask_pin = cnn_segment(
                classes_pin,
                path_i,
                plot=False,
                loaded_image=img,
                weights=True,
                weights_path=path_w_pin,
                model=model,
            )
            img_pin = colour_pin_mask(mask_pin, img)
            details_dict[i]["mask_p"] = img_pin
    else:
        img_pin = []

        del model

    if multiple_images == True:
        return details_dict
    else:
        return img, label, img_pin


def transform_labels(details_dict, with_pin=True):
    all_images, transformed, merged, orig_transformed = find_aligned_label(
        None,
        details_dict,
        None,
        paths=False,
        remove_pin=with_pin,
        template_option=2,
        ydist=True,
        align=True,
        combine_extra_masks=True,
        template_label=None,
        merge_method=0,
        filter_imgs=True,
    )

    try:
        aligned = align_merged_label(merged)
    except:
        try:
            aligned = adjust_alignment(merged, return_col=True)
        except:
            aligned = deepcopy(merged)

    return all_images, transformed, merged, orig_transformed, aligned


def ALICE_demo(
    paths, with_pin=True, pin_path=path_w_pin, label_path=path_w_lab, model=None
):
    if (with_pin == False) and (model == None):
        model = load_label_model(label_path)

    details_dict = get_masks_(
        paths,
        with_pin=with_pin,
        multiple_images=True,
        pin_path=pin_path,
        label_path=label_path,
        model=model,
    )
    all_images, transformed, merged, orig_transformed, aligned = transform_labels(
        details_dict, with_pin=with_pin
    )
    return all_images, transformed, merged, orig_transformed, aligned
