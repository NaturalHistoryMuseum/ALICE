import luigi
from pathlib import Path
import numpy as np
import cv2

from alice.config import (
    PROCESSING_MASK_DIR, 
    VISUALISATION_DIR,
    logger,
    DEBUG
)
from alice.predict import predict_masks
from alice.visualise import visualise_mask
from alice.tasks.base import BaseTask
from alice.tasks.image import ImageTask

class MaskRCNNTask(BaseTask):
    path = luigi.PathParameter()  
    
    def requires(self):
        return ImageTask(path=self.path)
    
    def output(self): 
        file_name = f'{self.path.stem}.npy'
        return luigi.LocalTarget(PROCESSING_MASK_DIR / file_name)    
    
    def run(self): 
        p = Path(self.input().path)
        predictions = predict_masks(p)          
        logger.info('%d labels detected in image %s', len(predictions['instances']), p.name)  
        mask = predictions["instances"].to("cpu").pred_masks
        np.save(self.output().path, mask.numpy())
        
        if DEBUG: # In debug mode, output a visulisation of the masks
            v = visualise_mask(p, predictions)
            cv2.imwrite(str(VISUALISATION_DIR / f'{p.stem}.jpg'), v)