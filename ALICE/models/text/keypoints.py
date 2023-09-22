import cv2
import numpy as np
from typing import Literal


class Keypoints():
    """
    Class for feature keypoints and descriptors
    """
    
    detector = cv2.FastFeatureDetector_create()
    # 5.00 scale factor for fast
    descriptor = cv2.xfeatures2d.BEBLID_create(5.00)
    
    def __init__(self, line):
        self.line = line
        self.image = line.image.copy()
        self.kps, self.desc = self._detect_and_compute(line)

    def _detect_and_compute(self, line):
        keypoints = self.detector.detect(line.image, None)        
        # FIXME: We've changed the line, so need to check the mask
        # keypoints = self._filter_keypoints(line.get_mask(), keypoints)     
        keypoints, desc = self.descriptor.compute(line.image, keypoints)        
        return keypoints, desc

    @staticmethod
    def _filter_keypoints(mask, keypoints):
        """
        Remove any keypoints that aren't part of the mask - the image part, not
        the white background revealed in deskewing
        """
        return [kp for kp in keypoints if mask[int(kp.pt[1]), int(kp.pt[0])] > 0]  
        
    @property
    def points(self) -> list:
        return self._to_points([kp.pt for kp in self.kps])

    @property
    def area(self) -> int:
        hull = cv2.convexHull(self.points)
        return cv2.contourArea(hull)

    def get_matched_points(self, matches, prop: Literal['queryIdx', 'trainIdx']):
        return self._to_points([self.kps[getattr(m, prop)].pt for m in matches])

    @staticmethod
    def _to_points(kp_pts):
        return np.float32(kp_pts).reshape(-1,1,2) 