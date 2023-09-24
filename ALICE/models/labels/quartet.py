import numpy as np
from scipy.stats import mode
from typing import List
import itertools
from collections import namedtuple 

from alice.config import logger
from alice.models.text import TextLineSegmentation, TextAlignment
from alice.models.labels.crop import CropLabels
from alice.models.labels.composite import Composite


QuartetResults = namedtuple('QuartetResults', ['labels', 'lines', 'composite'])


    
class LabelQuartet:
    """
    The four labels
    """

    def __init__(self, labels):
        self._labels = labels
        

    def process_labels(self):
        for i, label in enumerate(self._labels):
            logger.debug_image(label, f'cropped-{i}')  
                                      
        segmentations = [TextLineSegmentation(cropped) for cropped in self._labels]         
        segmentations = self._validate_textlines_per_label(segmentations)
        if segmentations:
            composite = Composite(segmentations)
            results = QuartetResults(self._labels, composite.composite_lines, composite.create_label())
        else:
            logger.info('Segmented labels do not have matching line numbers - no composite lines or labels')
            results = QuartetResults(self._labels, [], None)

        return results
        
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
        
        
            
        

            
            
            
        