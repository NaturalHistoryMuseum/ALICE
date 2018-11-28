import json
import numpy as np
import os
from PIL import Image
from mrcnn import config, utils
from skimage import draw


class LabelConfig(config.Config):
    NAME = 'label'
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + label
    STEPS_PER_EPOCH = 200
    DETECTION_MIN_CONFIDENCE = 0.9
    BACKBONE = 'resnet50'
    VALIDATION_STEPS = 10


class LabelDataset(utils.Dataset):
    class_name = 'label'

    def load_label(self, dataset_dir, subset):
        self.add_class(self.class_name, 1, self.class_name)
        assert subset in ['train', 'val']

        dataset_json = os.path.join(dataset_dir, subset, 'settings.json')
        with open(dataset_json, 'r') as jfile:
            content = json.load(jfile)

        dataset_dir = content['_via_settings']['core']['default_filepath']
        annotations = [a for a in content['_via_img_metadata'].values() if
                       a['regions'] and os.path.exists(
                           os.path.join(dataset_dir, a['filename']))]

        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]
            w, h = Image.open(os.path.join(dataset_dir, a['filename'])).size
            self.add_image(self.class_name, image_id=a['filename'],
                           path=os.path.join(dataset_dir, a['filename']),
                           width=w, height=h, polygons=polygons)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info['source'] != self.class_name:
            return super(LabelDataset, self).load_mask(image_id)

        mask = np.zeros([info['height'], info['width'], len(info['polygons'])],
                        dtype=np.uint8)
        for i, p in enumerate(info['polygons']):
            y = p['all_points_y']
            x = p['all_points_x']
            yi, xi = draw.polygon(y, x)
            mask[yi, xi, i] = 1

        bool_mask = mask.astype(bool)
        class_ids = np.ones([len(info['polygons'])], dtype=np.int32)
        return bool_mask, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] != self.class_name:
            return super(LabelDataset, self).image_reference(image_id)
        else:
            return info['path']
