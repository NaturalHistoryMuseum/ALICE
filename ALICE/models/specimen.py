import cv2
from pathlib import Path
from typing import List
from scipy.stats import mode

from alice.models.base import Base
from alice.models.mask import LabelMasks
from alice.models.label import Label, LabelValid, InvalidLabel
from alice.models.view import AngledView
from alice.models.quartet import LabelQuartet
from alice.config import PROCESSING_IMAGE_DIR, logger
from alice.utils import min_max
from alice.utils.image import save_image
from alice.log import init_log

class Specimen:
    def __init__(self, specimen_id, paths: List[Path]):
        init_log(specimen_id)
        self.specimen_id = specimen_id
        self.paths = paths
        
    def get_views(self):
        views = [AngledView(path, i)  for i, path in enumerate(self.paths)]
        assert len(views) == 4
        return views
            
    def get_label_quartets(self):
        views = self.get_views()
        modal = mode([len(view.labels) for view in views]) 
        self._validate_num_labels_per_view(views, modal.mode)          
        quartets = {}
        for label_index in range(0, modal.mode):
            quartet = LabelQuartet()
            for view in views:
                label = view.labels.get(label_index, InvalidLabel())
                quartet.add_label(label)     
            quartets[label_index] = quartet              
            logger.info('%s valid labels in level %s quartet', quartet.count_valid(), label_index)
            
        return quartets
        
    def get_labels(self):
        quartets = self.get_label_quartets()
        for level, quartet in quartets.items():               
            label_images = quartet.get_cropped_labels()
            logger.info('After size normalisation %s valid labels in level %s quartet', len(label_images), level)     
            for i, label_image in label_images.items():
                logger.debug_image(label_image, f'label-{level}-{i}')

            
                
        # FIXME - Add in getting the image bit
            
    @staticmethod
    def _validate_num_labels_per_view(views, mode):
        for view in views:
            if len(view.labels) != mode:
                lower_labels = list(view.labels.keys())[1:]
                for lower_label in lower_labels:
                    view.labels[lower_label].set_valid(LabelValid.NONDETECTED_LABELS)     
                
        
if __name__ == "__main__":

    # logger.set_specimen_id('011250151')
    # path = PROCESSING_IMAGE_DIR / '011250151_additional(1).JPG'    
    # view = SpecimenAngledView(path)    
    
    # specimen_id = '011245996'
    
    # paths = [PROCESSING_IMAGE_DIR / f'011245996_additional_{i}.jpeg' for i in range(1,5)]
    
    # # paths = [PROCESSING_INPUT_DIR / f'Tri434014_additional_{i}.JPG' for i in range(1,5)]
    specimen_id = '011250151'    
    paths = [PROCESSING_IMAGE_DIR / f'011250151_additional({i}).jpg' for i in range(1,5)]
    # specimen_id = 'Tri434014'    
    # paths = [PROCESSING_IMAGE_DIR / f'Tri434014_additional_{i}.jpg' for i in range(1,5)]    
    specimen = Specimen(specimen_id, paths)
    labels = specimen.get_labels()