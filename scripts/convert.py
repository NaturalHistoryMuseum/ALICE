import json
import typer
import cv2
import itertools
import numpy as np
from pathlib import Path
from detectron2.structures import BoxMode
from detectron2.data.datasets import convert_to_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog

from alice.config import logger

def main(dataset_path: Path):
    """    
    Script to convert maskrcnn dataset into coco & detectron data formats.
    
    e.g. to_coco.py data/balloon

    Arguments:
        dataset_path {Path} -- path to maskrcnn dataset
    """
    DatasetCatalog.clear()
    subdirs = [p for p in dataset_path.iterdir() if p.is_dir()]
    for subdir in subdirs:
        logger.info('Processing %s', subdir)
        dataset_dicts = []
        
        json_file = subdir / "via_region_data.json"
        with json_file.open('r') as f:
            images = json.load(f)
                        
        for image in images.values():
            record = {}
            file_path = subdir / image["filename"]
            height, width = cv2.imread(str(file_path)).shape[:2]
            
            record["file_name"] = file_path.name
            record["height"] = height
            record["width"] = width  
            # # image ID required for evaluators 
            # record["image_id"] = str(hash(file_path.name))            
            record["annotations"] = []  
            regions = image["regions"].values() if isinstance(image["regions"], dict) else image["regions"]

            for region in regions:
                # assert not region["region_attributes"]
                shape_attributes = region["shape_attributes"]
                px = shape_attributes["all_points_x"]
                py = shape_attributes["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]                
                poly = list(itertools.chain.from_iterable(poly))
                
                annotation = {
                    "bbox": [float(np.min(px)), float(np.min(py)), float(np.max(px)), float(np.max(py))],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                    "iscrowd": 0
                }
                
                record["annotations"].append(annotation)
            
            dataset_dicts.append(record)
    
        logger.info('Writing %s records to %s/detectron.json', len(dataset_dicts), subdir)
        with (subdir / 'detectron.json').open('w') as f:
            f.write(json.dumps(dataset_dicts, indent=4))
            
        dataset_name = "label/" + subdir.name
        DatasetCatalog.register(dataset_name, lambda: dataset_dicts)
        MetadataCatalog.get(dataset_name).set(thing_classes=["label"])

        logger.info('Writing %s records to %s/coco.json', len(dataset_dicts), subdir)
        convert_to_coco_json(dataset_name, str(subdir / 'coco.json'), allow_cached=False)

if __name__ == "__main__":
    typer.run(main)