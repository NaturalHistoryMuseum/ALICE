from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from pathlib import Path
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from pathlib import Path
import cv2
from numpy.typing import NDArray

from alice.config import MODEL_DIR


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = str(MODEL_DIR / "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set testing threshold
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (label)

mask_predictor = DefaultPredictor(cfg)  
    
    