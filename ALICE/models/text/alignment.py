import cv2
from typing import List
from skimage import metrics
import numpy as np

from alice.models.text.keypoints import Keypoints
from alice.models.text.line import TextLine
from alice.config import logger
from alice.utils.image import overlay_image

class TextAlignment():

    """
    Alignment of a collection of lines
    """
    
    # Transformed images much have at least this structural similarity to ref image
    ssim_threshold = 0.5
    # Must have at least this inlier threshold (inlier to src points)
    inlier_ratio_threshold = 0.4

    def __init__(self, lines: List[TextLine]):
        self.lines = lines        
        self.transformed_images = self._get_transformed_images()
        self.composite = self._create_composite()

    def _get_transformed_images(self):        
        lines_keypoints = [Keypoints(line) for line in self.lines]
        reference_idx = self._get_reference_image(lines_keypoints)           
        logger.info('Using line %s as reference', reference_idx)        
        ref_keypoints = lines_keypoints[reference_idx]
        other_keypoints = lines_keypoints[:reference_idx] + lines_keypoints[reference_idx+1:]
        transformed_images = [ref_keypoints.image]
        for keypoints in other_keypoints:          
            transformed = self._transform(ref_keypoints, keypoints)
            if transformed is None: continue
            ssim_score = self._calculate_ssim_score(ref_keypoints.image, transformed)
            if ssim_score < self.ssim_threshold:
                logger.info('Image does not meet SSIM score threshold')
            else:            
                transformed_images.append(transformed)
                
            overlayed = overlay_image(ref_keypoints.image.copy(), transformed)
            logger.debug_image(overlayed, f'transformed-line')

        return transformed_images

    @staticmethod
    def _calculate_ssim_score(ref_image, transformed):
        ref_grey = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        grey = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        return metrics.structural_similarity(ref_grey, grey, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
    
    def _get_reference_image(self, lines_keypoints):
        """
        Select the best reference image - calculate the hull around all of the keypoints
        The one with the largest hull is the best reference image
        """
        # Rather than calculating the area ratio based on image size
        # Just use area - so larger images are preferred
        keypoint_area = [kp.area for kp in lines_keypoints]
        return np.argmax(np.array(keypoint_area))

    def _match_descriptors(self, desc, ref_desc):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc, ref_desc)
        matches = sorted(matches, key = lambda x:x.distance)
        return matches

    def _transform(self, ref_keypoints, keypoints):           
        matches = self._match_descriptors(keypoints.desc, ref_keypoints.desc)
        src_pts = keypoints.get_matched_points(matches, 'queryIdx')
        dst_pts = ref_keypoints.get_matched_points(matches, 'trainIdx')
        
        M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        inlier_ratio = np.sum(inliers) / len(src_pts)
        if inlier_ratio < self.inlier_ratio_threshold:
            logger.info('Inlier ratio %s below threshold of %s - not performing transform', inlier_ratio, self.inlier_ratio_threshold)
            return
            
        h,w = ref_keypoints.image.shape[:2]
        transformed =  cv2.warpPerspective(keypoints.image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))       
        return transformed
        
    def _create_composite(self):        
        I = np.median(np.stack(self.transformed_images), axis=0)
        composite = np.array(I, dtype="uint8") 
        logger.debug_image(composite, f'composite-line')
        return composite  
    