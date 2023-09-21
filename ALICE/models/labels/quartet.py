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
            composite = Composite(segmentations)
            results = QuartetResults(cropped_labels, composite.composite_lines, composite.create_label())
        else:
            logger.info('Segmented labels do not have matching line numbers - no composite lines or labels')
            results = QuartetResults(cropped_labels, [], None)

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
        
        
            
        

            
            
            
        