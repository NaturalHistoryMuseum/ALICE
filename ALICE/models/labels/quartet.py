import numpy as np
from scipy.stats import mode
from typing import List
import itertools
from collections import namedtuple 

from alice.config import logger
from alice.models.text import TextLineSegmentation, TextAlignment
from alice.models.labels.crop import CropLabels



QuartetResults = namedtuple('QuartetResults', ['labels', 'lines', 'composite'])


    
class LabelQuartet:
    """
    The four labels
    """

    def __init__(self):
        self._labels = []
        
    def add_label(self, label):
        self._labels.append(label)
            
    def process_labels(self):
        cropped_labels = CropLabels(self._labels).crop()
        
        # FIXME: We can add another similarity comparison here??
                
        for i, cropped_label in enumerate(cropped_labels):
            logger.debug_image(cropped_label, f'cropped-{i}')        
        
        segmentations = [TextLineSegmentation(cropped) for cropped in cropped_labels]         
        segmentations = self._validate_textlines_per_label(segmentations)
        if segmentations:        
            composite_lines = self._get_composite_lines(segmentations)
            composite_label = self._merge_composites(composite_lines) if composite_lines else None
        else:
            logger.info('Segmented labels do not have matching line numbers - no composite lines or labels')
            composite_lines = []   
            composite_label = None    
        
        return QuartetResults(cropped_labels, composite_lines, composite_label)
        
    def _merge_composites(self, composite_lines):
        shape = np.array([comp.shape[:2] for comp in composite_lines])
        # Height is the sum of all the composites; width just the max
        height = np.sum(shape[:,0])
        width = np.max(shape[:,1])
        
        merged = np.full((height, width, 3), (255,255,255), dtype=np.uint8)

        y = 0
        for composite in composite_lines:
            h, w = composite.shape[:2]
            x = 0
            # Slice the filled composite into the image
            merged[y:y+h, x:x+w] = composite
            y+=h
            
        return merged
 
    @staticmethod
    def _validate_textlines_per_label(segmentations):
        """
        Filter segmentations which don't have the keys present in all the others
        """        
        line_keys = set(itertools.chain(*[segm.text_lines.keys() for segm in segmentations]))
        norm_segmentations = [segm for segm in segmentations if set(segm.text_lines.keys()) == line_keys]
        if len(norm_segmentations) != len(segmentations):
            logger.info("Not all labels have matching lines. %s of %s labels will be used", len(norm_segmentations), len(segmentations))
            
        return norm_segmentations    
    
    @staticmethod
    def _get_composite_lines(segmentations: List[TextLineSegmentation]) -> list:
        """
        Loop through the lines, getting them all from the different segmentations, aligning them and retrieveing 
        The compound
        """
        composites = []
        # We know they all have the same keys - so just grab the first
        line_keys = segmentations[0].text_lines.keys()
        for key in line_keys:
            lines = [segm.text_lines[key] for segm in segmentations]
            # FIXME: Do we want to debug each line? This is the place to do it
            alignment = TextAlignment(lines)
            composites.append(alignment.composite)
        return composites       
        
            
        

            
            
            
        