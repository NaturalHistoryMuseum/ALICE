from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


from alice.config import MODEL_DIR, EVAL_DIR


def evaluate():    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(MODEL_DIR / "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set testing threshold

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator('val', cfg, False, output_dir=str(EVAL_DIR))
    val_loader = build_detection_test_loader(cfg, 'val')
    result = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    