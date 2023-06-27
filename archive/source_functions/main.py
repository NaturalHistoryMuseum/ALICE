import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu
from skimage import measure
from copy import deepcopy
import cv2
import imutils
from imutils.perspective import four_point_transform
import pytesseract

try:
    from source_functions.alignment_helper_functions import *
    from source_functions.label_transformation import *
    from source_functions.label_merging import *
except:
    from alignment_helper_functions import *
    from label_transformation import *
    from label_merging import *


"""
Main functions used for ALICE software. Steps include:
1. Segment aligned labels from images.
2. LMerge labels.      
"""


def transform_label(
    img_path,
    msk_pth,
    mask_pin=None,
    remove_pin=True,
    pin_removal_method=0,
    template_label=None,
    dimension=None,
    combine_extra_masks=False,
    ydist=False,
    reconfig_method=0,
    paths=True,
    label_method=1,
    min_corner_angle=10,
    use_backup=0,
    min_size=20,
):
    # Aim: transform a label on an image of a pinned specimen to a segmented, aligned image.

    # Input: paths to original image / CNN mask. Optional: pin mask (in case you want to remove the pin), and template (additional alignment).
    # Output: transformed label image.

    # 1) Find corners of label mask:
    mask, img_orig, corners_x, corners_y, contours = find_corners(
        msk_pth,
        img_path,
        ydist=ydist,
        combine_extra_masks=combine_extra_masks,
        paths=paths,
        label_method=label_method,
    )

    # 2) Optional pin removal:
    if remove_pin == True:
        if pin_removal_method == 0:
            # 2.1) Convert pin mask to binary:
            mask_pin_bin = np.full((np.shape(mask)), False)
            mask_pin_bin[np.where(mask_pin == [255, 0, 0])[:2]] = True
            # 2.2) Test whether the pin mask coincides with the label mask:
            intersection = np.where((mask == True) & (mask_pin_bin == True))
            if len(intersection) == 0:
                img_updated = deepcopy(img_orig)
            else:
                img_updated = deepcopy(img_orig)
                img_updated[intersection] = [255, 255, 255]
        else:
            img_updated = hide_pin(img_orig, mask, mask_pin)
    else:
        img_updated = deepcopy(img_orig)

    # 3) Define short sides and long sides:
    short_inds, long_inds, _, dists_dict = define_label_sides(corners_x, corners_y)

    # 3.1) Check corners:
    cc = check_corners(
        corners_x, corners_y, short_inds, long_inds, min_angle=min_corner_angle
    )
    # 3.2) If corners don't fit criteria, redo the corner computation:
    if (cc == True) or (use_backup == 1):
        corners_x, corners_y = backup_corner_method(contours)
        short_inds, long_inds, _, dists_dict = define_label_sides(corners_x, corners_y)

    # 4) Update corners of parallelogram:
    corners_x_updated, corners_y_updated = reconfigure_corner_global(
        short_inds,
        long_inds,
        dists_dict,
        corners_x,
        corners_y,
        method=reconfig_method,
        original_mask=mask,
        original_image=img_orig,
    )
    # If these new corners are significantly larger than the original corners, we
    # use the original corners instead. Check this with compare_corners function:
    corners_x_updated, corners_y_updated = compare_corners(
        img_orig, corners_x, corners_y, corners_x_updated, corners_y_updated
    )

    # 5) Perspective transformation:
    if dimension == None:
        fixed_dim = False
    else:
        fixed_dim = True

    img_warped, _ = perspective_transform(
        corners_x_updated,
        corners_y_updated,
        img_updated,
        dists_dict,
        short_inds,
        long_inds,
        fixed_dim=fixed_dim,
        dimension=dimension,
    )

    if any(np.array(np.shape(img_warped)[:2]) < min_size) and (
        (use_backup == 0) or cc == False
    ):
        # If it's too small, repeat the process with an improved label:
        corners_x, corners_y = backup_corner_method(contours)
        short_inds, long_inds, _, dists_dict = define_label_sides(corners_x, corners_y)
        corners_x_updated, corners_y_updated = reconfigure_corner_global(
            short_inds,
            long_inds,
            dists_dict,
            corners_x,
            corners_y,
            method=reconfig_method,
            original_mask=mask,
            original_image=img_orig,
        )
        img_warped, _ = perspective_transform(
            corners_x_updated,
            corners_y_updated,
            img_updated,
            dists_dict,
            short_inds,
            long_inds,
            fixed_dim=fixed_dim,
            dimension=dimension,
        )

    # 6) Align with template:
    img_T = deepcopy(img_warped)
    if template_label != None:
        img_T = align_image(img_warped, template_label)

    return img_orig, img_warped, img_T, [corners_x_updated, corners_y_updated]


def find_aligned_label(
    img_name,
    path_dict,
    pin_masks=None,
    filter_opt=1,
    template_option=1,
    reconfig_method=0,
    remove_pin=True,
    pin_removal_method=0,
    align=False,
    template_label=None,
    angle_alignment=True,
    merge_method=0,
    filter_imgs=True,
    combine_extra_masks=False,
    ydist=False,
    paths=True,
    label_method=1,
    min_corner_angle=10,
    use_backup=0,
    test_stages=0,
    aligned_filter=False,
    backup_match_bound=0.95,
):
    # Aim: create a clear image of a specimen label.

    # Input: specimen name and dictionary of mask / image paths.
    # Output: segmented labels and merged label.

    """
    template_options: method in selecting a template from the images of a specimen.
    this template is used to align the other images to, before the merging step.
    Options:

    0 - first image.
    1 - largest image (label mask)
    2 - result from select_template function
    """

    if template_label != None:
        align = True

    # 1) Filter dataframe for img_name:
    if paths == True:
        tst = path_dict[path_dict["Position_ID"] == img_name].reset_index()
    all_orig_images = []
    all_transformed_images = []
    all_corners = []
    for i in range(0, 4):
        # 2) Get paths:
        if paths == True:
            msk_pth = tst.iloc[i]["Mask_Path"]
            img_pth = tst.iloc[i]["Image_Path"]
        else:
            msk_pth = deepcopy(path_dict[i]["mask_l"])
            img_pth = deepcopy(path_dict[i]["image"])

        # 3) Get pin mask:
        if remove_pin == True:
            if paths == True:
                mask_pin = pin_masks[tst["Image"].iloc[i]]
            else:
                mask_pin = deepcopy(path_dict[i]["mask_p"])
        else:
            mask_pin = None
        # 4) Transform label:
        if (i == 0) or (align == True):
            img_orig, img_warped, img_T, corners_ = transform_label(
                img_pth,
                msk_pth,
                mask_pin,
                remove_pin=remove_pin,
                pin_removal_method=pin_removal_method,
                template_label=template_label,
                combine_extra_masks=combine_extra_masks,
                ydist=ydist,
                paths=paths,
                label_method=label_method,
                min_corner_angle=min_corner_angle,
                use_backup=use_backup,
                reconfig_method=reconfig_method,
            )
        else:
            a, b, _ = np.shape(all_transformed_images[0])
            img_orig, img_warped, img_T, corners_ = transform_label(
                img_pth,
                msk_pth,
                mask_pin,
                remove_pin=remove_pin,
                pin_removal_method=pin_removal_method,
                template_label=template_label,
                dimension=(b, a),
                combine_extra_masks=combine_extra_masks,
                ydist=ydist,
                paths=paths,
                label_method=label_method,
                min_corner_angle=min_corner_angle,
                use_backup=use_backup,
                reconfig_method=reconfig_method,
            )

        all_orig_images.append(img_orig)
        all_transformed_images.append(img_T)
        all_corners.append(corners_)

    orig_transformed = deepcopy(all_transformed_images)

    # If you want to align the label but don't have a template, align all with the first label.
    if (align == True) and (template_label == None):
        if template_option == 2:
            try:
                template_label, all_transformed_images = select_template(
                    all_transformed_images
                )
            except:
                template_label = all_transformed_images[0]
        else:
            if template_option == 1:
                k = np.argmax(
                    [np.prod(np.shape(v)[:2]) for v in all_transformed_images]
                )
                template_label = all_transformed_images[k]
            else:
                template_label = all_transformed_images[0]

        if angle_alignment == True:
            opt_angle, _, _, _, _ = adjust_alignment(
                template_label, np.linspace(-10, 10, 21)
            )
            template_label = imutils.rotate_bound(template_label, opt_angle)
        aligned_labels = []
        aligned_labels_for_testing = []
        for ii, im in enumerate(all_transformed_images):
            try:
                img_T = align_image(im, template_label)
                aligned_labels.append(img_T)
                aligned_labels_for_testing.append(img_T)
            except:  # *temp fix*: if you can't align a label, ignore it.
                aligned_labels_for_testing.append(np.zeros((np.shape(im))))
                pass
        all_transformed_images = deepcopy(aligned_labels)
    elif (
        align == False
    ):  # if no template exists, let the first transformed label the template.
        template_label = all_transformed_images[0]

    # Filter or redo badly transformed images.
    if aligned_filter == True:
        new_transformed_images = []
        for tI in all_transformed_images:
            # Check if has any text.
            ocr_res = check_ocr(tI)
            if len(ocr_res) > 0:
                new_transformed_images.append(tI)
            else:
                img_T = align_image(
                    tI, template_label, matches_bound=backup_match_bound
                )
                ocr_res = check_ocr(img_T)
                if len(ocr_res) > 0:
                    new_transformed_images.append(img_T)
        all_transformed_images_orig = deepcopy(all_transformed_images)
        all_transformed_images = deepcopy(new_transformed_images)
    # 5) Merge labels:
    if merge_method == "both":
        merged1 = merge_label(all_transformed_images, method=0)
        merged2 = merge_label(all_transformed_images, method=1)
        merged = [merged1, merged2]
    else:
        if filter_imgs == True:
            try:
                if filter_opt == 1:

                    pix_diffs = [
                        np.sum(abs(template_label - I)) for I in all_transformed_images
                    ]
                    pix_diffs_ = [b for b in pix_diffs if b > 0]
                    ub = np.median(pix_diffs_) + (
                        np.percentile(pix_diffs_, 75) - np.percentile(pix_diffs_, 25)
                    )

                else:
                    merged = merge_label(all_transformed_images, method=merge_method)
                    pix_diffs = [
                        np.sum(abs(merged - I)) for I in all_transformed_images
                    ]
                    ub = np.median(pix_diffs) + 1.3 * (
                        np.percentile(pix_diffs, 75) - np.percentile(pix_diffs, 25)
                    )

                merged = merge_label(
                    [
                        I
                        for i, I in enumerate(all_transformed_images)
                        if pix_diffs[i] < ub
                    ],
                    method=merge_method,
                )
            except:
                try:
                    merged = merge_label(all_transformed_images, method=merge_method)
                except:
                    try:
                        sizes = [
                            np.shape(o_i)[0] * np.shape(o_i)[1]
                            for o_i in orig_transformed
                        ]
                        lb_ = np.average(sizes) / 3
                        size_y = [
                            np.shape(o_i)[0]
                            for ind_, o_i in enumerate(orig_transformed)
                            if (np.sum(o_i) > 0) and (sizes[ind_] >= lb_)
                        ]
                        size_x = [
                            np.shape(o_i)[1]
                            for ind_, o_i in enumerate(orig_transformed)
                            if (np.sum(o_i) > 0) and (sizes[ind_] >= lb_)
                        ]
                        sy = min(size_y)
                        sx = min(size_x)
                        all_transformed_images = [
                            o_i[:sy, :sx] for o_i in orig_transformed
                        ]
                        img_filtered = [
                            I
                            for ind_, I in enumerate(all_transformed_images)
                            if (np.sum(I) > 0) and (sizes[ind_] >= lb_)
                        ]
                        merged = merge_label(img_filtered, method=merge_method)
                    except:
                        merged = np.zeros((np.shape(template_label)))

        else:
            merged = merge_label(all_transformed_images, method=merge_method)
    if test_stages == 0:
        return all_orig_images, all_transformed_images, merged, orig_transformed
    else:
        return (
            all_orig_images,
            aligned_labels_for_testing,
            merged,
            orig_transformed,
            all_corners,
        )
