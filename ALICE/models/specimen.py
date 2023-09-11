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
from alice.log import clear_image_log

class Specimen:
    def __init__(self, specimen_id, paths: List[Path]):
        logger.set_specimen_id(specimen_id)
        clear_image_log(specimen_id)
        self.specimen_id = specimen_id
        self.paths = paths
        
    def get_views(self):
        views = []
        for i, path in enumerate(self.paths):
            view = SpecimenAngledView(path, i) 
            views.append(view)
            
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
                label = self._view_get_label(view, label_index)
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
    
    def __init__(self, path: Path, view_index):
        self.view_index = view_index
        self.path = path
        
        if not path.exists():
            raise FileNotFoundError(path)
                
        image = cv2.imread(str(path))
        super().__init__(image)   
                     
        self.label_masks = self.mask_image()
            
    def get_labels(self):
        labels = []
        for label_index, label_mask in enumerate(self.label_masks):                        
            masked_image = self.label_masks.image_with_higher_labels_masked(label_index)
            label = Label(label_mask, masked_image)
            labels.append(label)
            logger.debug_image(label.visualise(), f'view-{self.view_index}-label-{label_index}')
            logger.debug_image(label_mask.visualise(), f'view-{self.view_index}-labelmask-{label_index}')   
        return labels         

    def mask_image(self):
        label_masks = LabelMasks(self.image)
        logger.info(f'Detected %s labels masks in image %s', len(label_masks), self.path.name)
        logger.debug_image(label_masks.visualise(), f'view-mask-{self.view_index}')
        return label_masks        
                    
    def _visualise(self, image):
        return image    
        
if __name__ == "__main__":

    # logger.set_specimen_id('011250151')
    # path = PROCESSING_IMAGE_DIR / '011250151_additional(1).JPG'    
    # view = SpecimenAngledView(path)    
    
    # specimen_id = '011245996'
    
    # paths = [PROCESSING_IMAGE_DIR / f'011245996_additional_{i}.jpeg' for i in range(1,5)]
    
    # # paths = [PROCESSING_INPUT_DIR / f'Tri434014_additional_{i}.JPG' for i in range(1,5)]
    specimen_id = '011250151'
    
    paths = [PROCESSING_IMAGE_DIR / f'011250151_additional({i}).jpg' for i in range(1,5)]
    specimen = Specimen(specimen_id, paths)
    labels = specimen.get_labels()