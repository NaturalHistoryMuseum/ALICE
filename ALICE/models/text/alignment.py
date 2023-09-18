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

    @staticmethod
    def _calculate_reprojection_distance(src_pts, dst_pts, matrix):        
        """
        Calculate the mean distance between the transformed src points and the destination points 
        """
        transformed_points = cv2.perspectiveTransform(src_pts, matrix)
        reprojection_distance = np.linalg.norm(dst_pts - transformed_points.squeeze(), axis=1)
        return np.mean(reprojection_distance)      
    
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
        
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w = ref_keypoints.image.shape[:2]
        transformed =  cv2.warpPerspective(keypoints.image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))       

        # FIXME: Do I need this?
        reproj_dist = self._calculate_reprojection_distance(src_pts, dst_pts, M)
        print('Reproj dist:', reproj_dist)


        return transformed
        
    def _create_composite(self):
        
        images = [self._fill_whitespace(image) for image in self.transformed_images]
        
        I = np.median(np.stack(images), axis=0)
        composite = np.array(I, dtype="uint8") 
        logger.debug_image(composite, f'composite-line')
        return composite
    
    @staticmethod
    def _fill_whitespace(image):
        """
        Fill whitespace with mean background colour
        Need to do this before creating compound, so whitespace is filled/does not alter compound
        """
        # FIXME: REMOVE??
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create whiteapce mask for all pixels above 250
        whitespace_mask = grey > 240
        # And mask all text (to esclude from background colour calc)
        text_mask = grey > 200
        background_mask = ~whitespace_mask & text_mask
        # Get mean colour outside of text & whitespace
        colour = cv2.mean(image, mask=background_mask.astype(np.uint8))[:3]
        # And fill the whitespace
        image[whitespace_mask] = colour
        return image         
    