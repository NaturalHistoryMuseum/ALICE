import luigi
from pathlib import Path
from typing import List
import re

from alice.config import (
    INPUT_DIR, 
    logger
)

from alice.tasks.base import BaseTask
from alice.tasks.specimen import SpecimenTask

    
class ProcessTask(BaseTask):
        
    # Regex to pull out the specimen id from image file name
    # e.g. 011250151_additional(1) => specimen_id=011250151
    re_filename = re.compile(r'^(?P<specimen_id>[a-zA-Z0-9]+)_additional')
    limit = luigi.IntParameter(default=None)
    specimen_id = luigi.Parameter(default=None)
    
    def get_specimen_ids(self):
        paths = self.scan_directory()
        specimen_ids = set()
        for path in paths:
            specimen_id = self.parse_filename(path)
            specimen_ids.add(specimen_id)
        return list(specimen_ids)
    
    def scan_directory(self) -> List[Path]:
        image_extensions = ['.jpg', '.jpeg']
        image_paths = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in image_extensions]        
        return image_paths
    
    def parse_filename(self, path:Path):
        m = self.re_filename.match(path.stem)
        return m.group('specimen_id')
    
    def requires(self):
        """
        Loop through input dir, scheduling files 
        """
        specimen_ids = self.get_specimen_ids()
        
        if self.specimen_id:
            if self.specimen_id in specimen_ids: 
                logger.info('Queueing specimen %s for processing', self.specimen_id) 
                specimen_ids = [self.specimen_id]
            else:
                raise Exception(f'Specimen ID {self.specimen_id} set but not found in INPUT_DIR') 
        elif self.limit and self.limit < len(specimen_ids):
            logger.info('Queueing %s out of %s specimens for processing', self.limit, len(specimen_ids))  
            specimen_ids = specimen_ids[:self.limit]                
        else:
            logger.info('Queueing %s specimens for processing', len(specimen_ids)) 
                       
        for specimen_id in specimen_ids:
            yield SpecimenTask(specimen_id=specimen_id)

    
if __name__ == "__main__":
    # 012509422
    # 012509579
    
    luigi.build([ProcessTask(specimen_id='012509703')], local_scheduler=True)
    
