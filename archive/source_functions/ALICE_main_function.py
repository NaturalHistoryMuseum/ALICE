"""
ALICE SOFTWARE V2
-----------------

One function to obtain all merged label results from images obtained with ALICE.

This function and its steps, are based on the code laid out in all_steps.py.

"""

################################################################################

from all_steps import *


def get_merged_labels_from_ALICE(all_images):
    # Input: 4 images obtained with ALICE hardware of one pinned specimen.
    # Output: Merged versions of each label on specimen (up to a maximum number*), and transformed (aligned) labels from each image, for each label.
    # *the maximum number is set with variable MAX_NUMBER_OF_LABELS in all_steps.py.

    all_original_masks = []

    # Step 1: Get Images / Masks
    ############################

    for image in all_images:
        label_masks = segment_labels_from_image_path(image)
        all_original_masks.append(label_masks)

    # Step 2: Filter and Match Masks:
    #################################

    all_masks = match_labels_across_images(all_original_masks)

    """ The remaining steps loop through per group of masks e.g., if there were n labels found in the 4 images, 
    we go through the process n times, for each of the n'th labels in the image.
    """

    # Step 3 and 4: Find Corners and Warp Labels:
    ##############################################

    all_results = {}

    for n, n_labels_masks in enumerate(all_masks):
        warped_labels = []  # this list will contain the 4 warped masks of label n.

        for ind, label_mask in enumerate(n_labels_masks):
            image = all_images[ind]
            corner_results = define_box_around_label(label_mask, image)
            warped_label = warp_label(corner_results, image, label_mask)
            warped_labels.append(warped_label)

        # Step 5: Rotate Images:
        ########################
        rotated_labels = []
        binarized_labels = []
        for warped_label in warped_labels:
            rotated_label, binarized_label = adjust_and_rotate_warped_label(
                warped_label
            )
            rotated_labels.append(rotated_label)
            binarized_labels.append(binarized_label)

        # Step 6: Select Template:
        ##########################
        template_warped_label = select_template_warped_label(
            rotated_labels, binarized_labels
        )

        # Step 7: Align Labels to Template:
        ###################################
        aligned_labels = align_warped_labels(rotated_labels, template_warped_label)

        # Step 8: Filter Out Bad Labels:
        ################################
        filtered_labels = find_all_good_labels(aligned_labels, template_warped_label)

        # Step 9: Merge Labels:
        ################################
        merged_label = merge_aligned_labels(filtered_labels)

        # Step 10: Classify Label as Good or Bad:
        ########################################
        good_label_binary = is_label_good(merged_label)

        all_results.update(
            {
                n: {
                    "merged_label": merged_label,
                    "good_classification": good_label_binary,
                    "filtered_aligned_labels": filtered_labels,
                }
            }
        )

    return all_results


################################################################################

"""
-------------
EXAMPLE CODE
-------------
"""

# import cv2

# example_image_paths = [
#     "example_id_0.png",
#     "example_id_1.png",
#     "example_id_2.png",
#     "example_id_3.png",
# ]

# all_images = [cv2.imread(image_path) for image_path in example_image_paths]
# all_results = get_merged_labels_from_ALICE(all_images)
