from pathlib import Path
from typing import List
import re
import itertools
from collections import defaultdict
import imutils
from PIL import Image
import numpy as np


from alice.models.labels import AngledView, InvalidLabel, LabelQuartet
from alice.config import RESIZED_IMAGE_DIR, logger, NUM_CAMERA_VIEWS
from alice.log import init_log
from alice.models.labels.crop import CropLabels
from alice.utils.image import compute_image_hash_similarity
from alice.clip import CLIP

class Specimen:
    
    # Regex to pull out image id from file name
    # e.g. 011250151_additional(1) => image_id=1
    # Brackets around e.g. (1) are optional - will work for Tri434015_additional_4
    re_filename = re.compile(r'additional_?\(?(?P<image_idx>[1-4])\)?$')
    clip = CLIP()
    
    def __init__(self, specimen_id, paths: List[Path]):
        init_log(specimen_id)
        self.specimen_id = specimen_id
        self.paths = self._get_sorted_paths(paths)
        try:
            assert len(self.paths) == NUM_CAMERA_VIEWS
        except AssertionError as e:        
            raise(AssertionError(f"{len(self.paths)} images found"))
        
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
        return views
            
    def get_label_quartets(self):
        views = self.get_views()
        
        label_levels = self._group_cropped_labels_by_level(views)
        
        if not self._label_levels_are_similar_images(label_levels):
            logger.info('Label images across levels are disimilar - regrouping by similarity')        
            label_levels = self._regroup_labels_by_similarity(label_levels)
        else:
            logger.info('Label similarity check - passed.')  
            
        quartets = []
        for i, labels in label_levels.items():
            # If we only have one label in the cluster, remove any with max edge < 150
            # Just a fragment of a label 
            if len(labels) == 1:
                max_edge = max(labels[0].shape[:2])
                if max_edge < 150:
                    logger.info('Level %s: has 1 small label (max edge %s). Ignoring', i, max_edge)
            
            logger.info('Level %s: %s labels', i, len(labels))
            quartets.append(LabelQuartet(labels))                        
        return quartets
    
    def _group_cropped_labels_by_level(self, views):
        num_levels = max([len(view.labels) for view in views])
        label_levels = {}
        for level in range(0, num_levels):
            # If a label doesn't exist at a particular level for a view, we create InvalidLabel()
            # So that order of labels will be preserved in the crop & rotate
            label_levels[level] = CropLabels([view.labels.get(level, InvalidLabel()) for view in views]).crop() 
        return label_levels 
    
    def _label_levels_are_similar_images(self, label_levels):
        # Permissive similarity - hash comparison for mtaching images ~ 13
        # But we assume our initial label view arrangement is correct 
        threshold = 20    

        for labels in label_levels.values():
        
            image_list = [Image.fromarray(label) for label in labels]
            num_images = len(labels)    
            scores = []
            for i, j in itertools.combinations(range(num_images), 2):        
                image1 = image_list[i]
                image2 = image_list[j]            
                hash_similarity = compute_image_hash_similarity(image1, image2)
                scores.append(hash_similarity)

            is_similar = np.all(np.array(scores) < threshold)
            # If the images in a row aren't similar, we need to cluster the 
            if not is_similar:
                return False
            
        return True
    
    def _regroup_labels_by_similarity(self, label_levels):  
        all_labels = list(itertools.chain(*label_levels.values()))
        clusters = self.clip.cluster(all_labels)
        logger.info('%s clusters with similar images detected', len(clusters))        
        clustered_images = {}
        for i, cluster in enumerate(clusters):
            images = [all_labels[image_idx] for image_idx in cluster]
            logger.info('Cluster %s has %s matching images', i, len(images))            
            clustered_images[i] = images
        return clustered_images
        
    def process(self):
        quartets = self.get_label_quartets()
        all_results = {}
        for level, quartet in enumerate(quartets): 
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

if __name__ == "__main__":

    # logger.set_specimen_id('011250151')
    # path = RESIZED_IMAGE_DIR / '011250151_additional(1).JPG'    
    # view = SpecimenAngledView(path)    
    
    # specimen_id = '011244568'
    
    # paths = [RESIZED_IMAGE_DIR / f'011245996_additional_{i}.jpeg' for i in range(1,5)]
    
    # specimen_id = '011250151'    
    # paths = [RESIZED_IMAGE_DIR / f'011250151_additional({i}).jpg' for i in range(1,5)]
    # specimen_id = '011250151'   
    specimen_id = '012509519'
    specimen_id = 'Tri434015'        
    paths = [p.resolve() for p in RESIZED_IMAGE_DIR.glob(f'{specimen_id}*.*') if p.suffix.lower() in {".jpg", ".jpeg"}] 
    specimen = Specimen(specimen_id, paths)    
    labels = specimen.process()