from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
import json
import os


from alice.config import MODEL_DIR, TRAIN_DATASET_PATH, VAL_DATASET_PATH, NUM_EPOCHS


class COCOEvaluatorTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    

def calculate_max_iterations(num_images, batch_size, num_epochs):
    """
    To calculate max iterations:  (num_images / batch_size) * num_epochs

    Arguments:
        num_images
        batch_size
        num_epochs
    """    
    return round((num_images / batch_size) * num_epochs)

def train(num_epochs:int = NUM_EPOCHS):
    
    with (TRAIN_DATASET_PATH / "coco.json").open('r') as f:
        train_dataset = json.load(f)       
        num_images = len(train_dataset['images'])
            
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ('train',)
    cfg.DATASETS.TEST = ('val', )
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.OUTPUT_DIR = str(MODEL_DIR)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1                     
    cfg.SOLVER.MAX_ITER = calculate_max_iterations(num_images, cfg.SOLVER.IMS_PER_BATCH, num_epochs)
    
    register_coco_instances('train', {}, TRAIN_DATASET_PATH / "coco.json", TRAIN_DATASET_PATH)
    register_coco_instances('val', {}, VAL_DATASET_PATH / "coco.json", VAL_DATASET_PATH)    
                         
    # trainer = COCOEvaluatorTrainer(cfg)     
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()    


if __name__ == "__main__":
    typer.run(train)