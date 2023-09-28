import luigi
import cv2

from alice.config import IMAGE_BASE_WIDTH, RESIZED_IMAGE_DIR
from alice.tasks.base import BaseTask
from alice.utils.image import resize_image

class ImageTask(BaseTask):
    path = luigi.PathParameter()  
        
    def run(self):     
        image = cv2.imread(str(self.path))
        image = resize_image(image, IMAGE_BASE_WIDTH, IMAGE_BASE_WIDTH) 
        cv2.imwrite(self.output().path, image) 
                
    def output(self): 
        return luigi.LocalTarget(RESIZED_IMAGE_DIR / self.path.name if IMAGE_BASE_WIDTH else self.path.name)