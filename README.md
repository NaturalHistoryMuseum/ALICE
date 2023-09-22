# ALICE Software


### Installation

pip install -r requirements

#### Installation errors

###### Pytorhc is not installed.

Make sure wheel is installed and pip >= 23.2.1 before running pip install -r requirements


Code to extract labels from pinned specimen images.

Aims:
1. Find boundaries of labels with [Mask R-CNN](https://github.com/matterport/Mask_RCNN).
1. Approximate corners of labels.
1. Transform labels to a 2d viewpoint.
1. Align labels to a template.
1. Merge labels.


# Converting mask-rcnn region json files into coco/detectron JSON.

```
python scripts/convert.py data/label
```

The script will loop through child directories in data/label (data/label/var and data/label/train), read the via_region_data.json, and then output two files coco.json and detectron.json.

Train uses coco.json. Detectron.json is legacy code, and will be removed in future versions. 

via_region_data.json must be within the same directory as the images - e.g. data/label/var/via_region_data.json


# ALICE Module

Imports all use the alice module namespace - to activate for development:

```
python setup.py develop
```


