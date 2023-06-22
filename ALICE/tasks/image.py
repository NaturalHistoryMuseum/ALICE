import luigi
from PIL import Image
from alice.config import (
    PROCESSING_IMAGE_DIR,
    IMAGE_BASE_WIDTH,
    logger,
)
from alice.tasks.base import BaseTask


class ImageTask(BaseTask):
    path = luigi.PathParameter()  
        
    def run(self):     
        # Only runs if the image doesn't exist - being resized
        img = Image.open(self.path)
        w, h = img.size
        resize_height = int(h * float(IMAGE_BASE_WIDTH / w))     
        logger.info('Resizimg image %s to (%d, %d)', self.path.name, IMAGE_BASE_WIDTH, resize_height)       
        img = img.resize((IMAGE_BASE_WIDTH, resize_height), Image.Resampling.LANCZOS)
        img.save(self.output().path)            
                
    def output(self): 
        return luigi.LocalTarget(PROCESSING_IMAGE_DIR / self.path.name if IMAGE_BASE_WIDTH else self.path)