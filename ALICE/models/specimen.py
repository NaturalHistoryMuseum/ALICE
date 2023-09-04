import cv2
from pathlib import Path
from typing import List
from scipy.stats import mode

from alice.models.base import Base
from alice.models.mask import LabelMasks
from alice.models.label import Label, LabelValid, LabelQuartet, InvalidLabel
from alice.config import PROCESSING_IMAGE_DIR, logger
from alice.utils import min_max
from alice.utils.image import save_image

class Specimen:
    def __init__(self, specimen_id, paths: List[Path]):
        logger.set_specimen_id(specimen_id)
        self.specimen_id = specimen_id
        self.paths = paths
        
    def get_labels(self):
        views = [SpecimenAngledView(p) for p in self.paths]
        modal = mode([len(view.labels) for view in views]) 
        self._validate_num_labels_per_view(views, modal.mode)
        
        # quartets = []
        for label_index in range(0, modal.mode):
            quartet = LabelQuartet()
            for view in views:
                label = self._view_get_label(view, label_index)
                quartet.add_label(label)
            
            label_images = quartet.get_cropped_labels()
            
            for i, label_image in label_images.items():
                logger.debug_image(label_image, f'label-{self.specimen_id}-{label_index}-{i}')

            
        
        
        # FIXME - Add in getting the image bit
            
    @staticmethod
    def _validate_num_labels_per_view(views, mode):
        for view in views:
            if len(view.labels) != mode:
                for label in view.labels[1:]:
                    label.set_valid(LabelValid.NONDETECTED_LABELS)                
    @staticmethod          
    def _view_get_label(view, label_index):
        try:
            label = view.labels[label_index]
        except IndexError:
            # We do not have a label at this index position -
            # Will happen if there's no mask, but we want preserve
            # the order of other labels so insert InvalidLabel()
            label = InvalidLabel()
        return label        
        

    
class SpecimenAngledView(Base):

    """
    A view of the specimen 
    """
    
    def __init__(self, path: Path):
        image = cv2.imread(str(path))
        super().__init__(image)
        
        self.labels = []
        
        logger.info('Initiating labels masks')
        self.label_masks = LabelMasks(image)
        logger.info(f'Detected %s labels masks in image %s', len(self.label_masks), path.name)
        logger.debug_image(self.label_masks.visualise(), 'label-masks')
        
        for label_index, label_mask in enumerate(self.label_masks):
            image = self.label_masks.get_masked_image(label_index)
            logger.debug_image(image, f'masked-image-{label_index}')
            self.labels.append(Label(label_mask, image))  
            
    def _visualise(self, image):
        return image    
    
    
if __name__ == "__main__":

    # logger.set_specimen_id('011250151')
    # path = PROCESSING_IMAGE_DIR / '011250151_additional(1).JPG'    
    # view = SpecimenAngledView(path)    
    
    specimen_id = '011245996'
    
    paths = [PROCESSING_IMAGE_DIR / f'011245996_additional_{i}.jpeg' for i in range(1,5)]
    
    # # paths = [PROCESSING_INPUT_DIR / f'Tri434014_additional_{i}.JPG' for i in range(1,5)]
    # # specimen_id = '011250151'
    # # paths = [PROCESSING_IMAGE_DIR / f'011250151_additional({i}).JPG' for i in range(1,5)]
    specimen = Specimen(specimen_id, paths)
    labels = specimen.get_labels()