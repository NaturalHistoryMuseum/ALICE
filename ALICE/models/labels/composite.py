import numpy as np
import imutils
from typing import List, Iterable
from numpy.typing import NDArray

from alice.config import logger
from alice.models.text import TextLineSegmentation, TextAlignment


class Composite:
    
    def __init__(self, segmentations: List[TextLineSegmentation]):  
        self.composite_lines = self._get_composite_lines(segmentations)
        
    def _get_composite_lines(self, segmentations: List[TextLineSegmentation]) -> List[NDArray]:
        composites = []
        for i, lines in self._group_lines(segmentations):
            alignment = TextAlignment(lines)
            # for j, transformed_image in enumerate(alignment.transformed_images):
            #     logger.debug_image(transformed_image, f'tranformed-{i}-{j}')                 
            composites.append(alignment.composite)
        return composites
        
    @staticmethod
    def _group_lines(segmentations: List[TextLineSegmentation]) -> Iterable[tuple]:
        # We know they all have the same keys - so just grab the first
        line_keys = segmentations[0].text_lines.keys()
        for key in line_keys:
            lines = [segm.text_lines[key] for segm in segmentations]
            yield key, lines
            
            
    def create_label(self) -> NDArray:
        shape = np.array([line.shape[:2] for line in self.composite_lines])
        # Height is the sum of all the composites; width just the max
        height = np.sum(shape[:,0])
        width = np.max(shape[:,1])
        
        merged = np.full((height, width, 3), (255,255,255), dtype=np.uint8)

        y = 0
        for line in self.composite_lines:
            h, w = line.shape[:2]
            x = 0
            # Slice the filled line into the image
            merged[y:y+h, x:x+w] = line
            y+=h
            
        return merged            