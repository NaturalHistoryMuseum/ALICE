import luigi
from pathlib import Path
from typing import List
import re
import cv2

from alice.config import (
    PROCESSING_INPUT_DIR, 
    PROCESSING_NUM_CAMERA_VIEWS, 
    PROCESSING_OUTPUT_DIR,
    PROCESSING_IMAGE_DIR,
    IMAGE_BASE_WIDTH,
    logger
)

from alice.tasks.base import BaseTask
from alice.utils.image import resize_image
from alice.models.specimen import Specimen

class ImageTask(BaseTask):
    path = luigi.PathParameter()  
        
    def run(self):     
        image = cv2.imread(str(self.path))
        image = resize_image(image, IMAGE_BASE_WIDTH, IMAGE_BASE_WIDTH) 
        cv2.imwrite(self.output().path, image) 
                
    def output(self): 
        return luigi.LocalTarget(PROCESSING_IMAGE_DIR / self.path.name if IMAGE_BASE_WIDTH else self.path.name)
    
    
class SpecimenTask(BaseTask):
    # Regex to pull out image id from file name
    # e.g. 011250151_additional(1) => image_id=1
    # Brackets around e.g. (1) are optional - will work for Tri434015_additional_4
    re_filename = re.compile(r'additional_?\(?(?P<image_id>[1-4])\)?$')
    
    specimen_id = luigi.Parameter()
    
    def glob_image_paths(self):
        return [p.resolve() for p in Path(PROCESSING_INPUT_DIR).glob(f'{self.specimen_id}*.*') if p.suffix.lower() in {".jpg", ".jpeg"}]    

    def parse_filename(self, path:Path):
        m = self.re_filename.search(path.stem)
        return int(m.group('image_id'))

    def requires(self):
        
        image_paths = self.glob_image_paths()        
        num_image_paths = len(image_paths)
        if num_image_paths != PROCESSING_NUM_CAMERA_VIEWS:
            raise IOError(f'Only {num_image_paths} images found for {self.specimen_id}')
        
        for path in image_paths:
            yield ImageTask(path=path)

    def run(self):
        
        # images = {self.parse_filename()}
        images = {}
        for i in self.input():  
            path = Path(i.path)
            image_id = self.parse_filename(path)
            images[image_id] = path
            
        # Sort the dictionary items by their keys (image id), so the order is preserved
        sorted_images = sorted(images.items())
        # Sorted dict produces list of tuples - so get the path bits
        image_paths = [t[1] for t in sorted_images]
        
        specimen = Specimen(self.specimen_id, paths=image_paths)
        labels = specimen.get_labels()
   
                
        # print(image_paths)

        
    def output(self):     
        return luigi.LocalTarget(PROCESSING_OUTPUT_DIR / f'{self.specimen_id}-composite.jpg')
    
    
class ProcessTask(BaseTask):
        
    # Regex to pull out the specimen id from image file name
    # e.g. 011250151_additional(1) => specimen_id=011250151
    re_filename = re.compile(r'^(?P<specimen_id>[a-zA-Z0-9]+)_additional')
    
    def get_specimen_ids(self):
        paths = self.scan_directory()
        specimen_ids = set()
        for path in paths:
            specimen_id = self.parse_filename(path)
            specimen_ids.add(specimen_id)
        return specimen_ids
    
    def scan_directory(self) -> List[Path]:
        image_extensions = ['.jpg', '.jpeg']
        image_paths = [f for f in PROCESSING_INPUT_DIR.iterdir() if f.suffix.lower() in image_extensions]        
        return image_paths
    
    def parse_filename(self, path:Path):
        m = self.re_filename.match(path.stem)
        return m.group('specimen_id')
    
    def requires(self):
        """
        Loop through input dir, scheduling files 
        """
        specimen_ids = self.get_specimen_ids()
        logger.info('Queueing %s specimens for processing', len(specimen_ids))           
        for specimen_id in specimen_ids:
            yield SpecimenTask(specimen_id=specimen_id)

    
if __name__ == "__main__":
    # luigi.build([SpecimenTask(specimen_id='011244568')], local_scheduler=True)
    luigi.build([ProcessTask()], local_scheduler=True)
    
