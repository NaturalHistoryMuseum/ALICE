from pathlib import Path
from typing import List
from scipy.stats import mode
import re

from alice.models.labels import AngledView, LabelValid, InvalidLabel, LabelQuartet
from alice.models.text.alignment import TextAlignment
from alice.config import PROCESSING_IMAGE_DIR, logger, OUTPUT_DATA_DIR
from alice.log import init_log

class Specimen:
    
    # Regex to pull out image id from file name
    # e.g. 011250151_additional(1) => image_id=1
    # Brackets around e.g. (1) are optional - will work for Tri434015_additional_4
    re_filename = re.compile(r'additional_?\(?(?P<image_idx>[1-4])\)?$')
    
    def __init__(self, specimen_id, paths: List[Path]):
        init_log(specimen_id)
        self.specimen_id = specimen_id
        self.paths = self._get_sorted_paths(paths)
        
    def _parse_filename_idx(self, path:Path):
        m = self.re_filename.search(path.stem)
        return int(m.group('image_idx')) 
    
    def _get_sorted_paths(self, paths:List[Path]):       
        # Create a dict of paths, keyed by the index value in the file name
        paths_dict = {
            self._parse_filename_idx(path): path for path in paths
        }
        # Return list of paths, sorted by key
        return [paths_dict[key] for key in sorted(paths_dict.keys())]
        
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
        return quartets
        
    def process(self):
        quartets = self.get_label_quartets()
        all_results = {}
        for level, quartet in quartets.items(): 
            logger.info('Processing quartet level %s', level)
            logger.info('Quartet %s: has %s cropped labels for text detection', level, len(quartet._labels))
            
            results = quartet.process_labels()
            
            # Log some results
            logger.info('Quartet %s: %s labels cropped and processed', level, len(results.labels))
            for i, label_image in enumerate(results.labels):
                logger.debug_image(label_image, f'label-{level}-{i}')            
            
            logger.info('Quartet %s: %s text lines detected', level, len(results.lines))
            for i, lines in enumerate(results.lines):
                logger.debug_image(lines, f'lines-{level}-{i}')              
        
            if results.composite is None:      
                logger.info('Quartet %s: no composite image', level)    
            else:
                logger.debug_image(results.composite, f'composite-{level}') 
            
            all_results[level] = results

        return all_results
                
            
    @staticmethod
    def _validate_num_labels_per_view(views, mode):
        for i, view in enumerate(views):
            logger.info('View %s: %s labels', i, len(view.labels)) 
            if len(view.labels) != mode:
                lower_labels = list(view.labels.keys())[1:]
                for lower_label in lower_labels:
                    view.labels[lower_label].set_valid(LabelValid.NONDETECTED_LABELS) 

if __name__ == "__main__":

    # logger.set_specimen_id('011250151')
    # path = PROCESSING_IMAGE_DIR / '011250151_additional(1).JPG'    
    # view = SpecimenAngledView(path)    
    
    specimen_id = '011244568'
    
    # paths = [PROCESSING_IMAGE_DIR / f'011245996_additional_{i}.jpeg' for i in range(1,5)]
    
    # specimen_id = '011250151'    
    # paths = [PROCESSING_IMAGE_DIR / f'011250151_additional({i}).jpg' for i in range(1,5)]
    # specimen_id = '011250151'   
    # specimen_id = 'Tri434014'    

    paths = [p.resolve() for p in Path(PROCESSING_IMAGE_DIR).glob(f'{specimen_id}*.*') if p.suffix.lower() in {".jpg", ".jpeg"}] 
    specimen = Specimen(specimen_id, paths)
    labels = specimen.process()