import cv2
from pathlib import Path
from typing import List
from scipy.stats import mode
from collections import OrderedDict

from alice.models.base import Base
from alice.models.labels import LabelMasks, Label
from alice.log import init_log
from alice.config import RESIZED_IMAGE_DIR, logger


class AngledView(Base):

    """
    A view of the specimen 
    """
    
    def __init__(self, path: Path, view_index):
        self.view_index = view_index
        self.path = path
        
        if not path.exists():
            raise FileNotFoundError(path)
                
        image = cv2.imread(str(path))
        super().__init__(image)
        self.label_masks = self._mask_image()
        self.labels = self._get_labels()
            
    def _get_labels(self):
        labels = []
        for label_index, label_mask in enumerate(self.label_masks):                        
            masked_image = self.label_masks.image_with_higher_labels_masked(label_index)
            label = Label(label_mask, masked_image)
            labels.append((label_index, label))
            logger.debug_image(label.visualise(), f'view-{self.view_index}-label-{label_index}')
            # logger.debug_image(label_mask.visualise(), f'view-{self.view_index}-highermasks-{label_index}')   
            
        return OrderedDict(labels)         

    def _mask_image(self):
        label_masks = LabelMasks(self.image)
        logger.info(f'Detected %s labels masks in image %s', len(label_masks), self.path.name)
        logger.debug_image(label_masks.visualise(), f'view-{self.view_index}-mask')
        return label_masks        
                    
    def _visualise(self, image):
        return image   
    
    
    
if __name__ == "__main__":

    # 
    # path = PROCESSING_IMAGE_DIR / '011250151_additional(1).JPG'    
    # view = SpecimenAngledView(path)    
    
    # specimen_id = '011245996'
    
    # paths = [PROCESSING_IMAGE_DIR / f'011245996_additional_{i}.jpeg' for i in range(1,5)]
    
    # # paths = [PROCESSING_INPUT_DIR / f'Tri434014_additional_{i}.JPG' for i in range(1,5)]
    specimen_id = '011250151'
    
    # path = PROCESSING_IMAGE_DIR / f'Tri434014_additional_4.jpg'
    path = RESIZED_IMAGE_DIR / '011250151_additional(2).JPG'
    init_log(specimen_id)
    view = AngledView(path, 1)
 