import luigi
from pathlib import Path
import re

from alice.config import (
    PROCESSING_INPUT_DIR, 
    PROCESSING_NUM_CAMERA_VIEWS, 
    PROCESSING_OUTPUT_DIR,
)

from alice.tasks.base import BaseTask
from alice.tasks.mask_rcnn import MaskRCNNTask


class LabelAlignmentTask(BaseTask):
        
    image_id = luigi.Parameter()  
    # Regex to pull out the name (minus 1,2, etc.,) and ext
    re_filename = re.compile(r'(?P<name>.*)[0-9]{1}.(?P<ext>jpe?g)$')
    
    def glob_files(self):
        return [p.resolve() for p in Path(PROCESSING_INPUT_DIR).glob(f'{self.image_id}*.*') if p.suffix in {".jpg", ".jpeg"}]
    
    def parse_filename(self, path:Path):
        m = self.re_filename.search(path.name)
        return m.group('name'), m.group('ext')  
    
    def get_filename_and_extension(self):    
        files = self.glob_files()
        if not files: 
            raise IOError(f'No image files found for ID {self.image_id}') 
        names, exts = zip(*[self.parse_filename(f) for f in files])
        # Code would break if images don't have same name & extension 
        assert len(set(names)) == 1
        assert len(set(exts)) == 1    
        return names[0], exts[0]     
    
    def requires(self):
        """
        Loop through expected number of camera views, adding a photograph

        * :py:class:`~.PhotoTask`

        :return: list of object (:py:class:`luigi.task.Task`)
        """
        name, ext = self.get_filename_and_extension()        
        for i in range(1, PROCESSING_NUM_CAMERA_VIEWS + 1):
            yield MaskRCNNTask(path=PROCESSING_INPUT_DIR / f'{name}{i}.{ext}')

    def run(self):
        for i in self.input():  
            print(i)
        
    def output(self):     
        return luigi.LocalTarget(PROCESSING_OUTPUT_DIR / f'{self.image_id}-composite.jpg') 
    
if __name__ == "__main__":
    # luigi.build([ProcessSpecimenTask(image_id='011244568', force=True)], local_scheduler=True)
    luigi.build([LabelAlignmentTask(image_id='011244568', force=True)])