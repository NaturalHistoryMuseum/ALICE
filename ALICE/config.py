from pathlib import Path
import logging
from enum import Enum
import os
import cv2

DEBUG = os.getenv('DEBUG') or 1

INVALID_LABEL_SHAPE = -1

# Maximum number of labels to look at per image.
MAX_NUMBER_OF_LABELS = 6

###### Paths #######

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / 'data'
TEST_DIR = ROOT_DIR / 'test'
TEST_IMAGE_DIR = TEST_DIR / 'images'

ASSETS_DIR = DATA_DIR / 'assets'
MODEL_DIR = DATA_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_DIR = DATA_DIR / 'evaluation'
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT_DIR / '.cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT_DIR / 'log'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# VISUALISATION_DIR = Path(DATA_DIR / 'visualisations')
# VISUALISATION_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATA_DIR = Path(DATA_DIR / 'output')
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATASET_PATH = DATA_DIR / 'label' / 'train'
VAL_DATASET_PATH = DATA_DIR / 'label' / 'val'

IMAGE_BASE_WIDTH = 2048
PROCESSING_INPUT_DIR = Path(DATA_DIR / 'input')
PROCESSING_IMAGE_DIR = Path(DATA_DIR / 'images')
PROCESSING_OUTPUT_DIR = Path(DATA_DIR / 'output')
PROCESSING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSING_NUM_CAMERA_VIEWS = 4 

if IMAGE_BASE_WIDTH:
    resized_dir = f'{IMAGE_BASE_WIDTH}x'
    PROCESSING_IMAGE_DIR /= resized_dir

PROCESSING_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(DATA_DIR / 'models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)



###### Train config #######

NUM_EPOCHS = 1





# # Directory to save logs and model checkpoints, if not provided
# # through the command line argument --logs
# DEFAULT_LOGS_DIR = DATA_DIR / "logs"

###### Logging #######

# Define a custom log level named 'VERBOSE'
DEBUG_IMAGE = 15

# Add the custom log level to the logging module
logging.addLevelName(DEBUG_IMAGE, 'DEBUG_IMAGE')

class DebugLogger(logging.Logger):

    def __init__(self, name):
        super().__init__(name)
        self.specimen_id = None
    
    def set_specimen_id(self, specimen_id):
        self.specimen_id = specimen_id

    def debug_image(self, image, debug_code=None):
        
        if not self.specimen_id:
            self.critical(f"No specimen ID set in DebugLogger")
            return
            
        if self.isEnabledFor(DEBUG_IMAGE):
            dir_path = LOG_DIR / self.specimen_id
            dir_path.mkdir(parents=True, exist_ok=True)
            file_name = f'{debug_code}.jpg'
            path = dir_path / file_name
            if path.exists():
                existing_file_count = len(list(dir_path.glob(f'{file_name}*')))                
                path = dir_path / f'{path.stem}-{existing_file_count}.jpg'
                        
            cv2.imwrite(str(path), image) 
            
# Reset default logging levels            # 
# logging.basicConfig(level=logging.NOTSET)        
# Set up logging - inherit from luigi so we use the same interface
logger = DebugLogger("luigi-interface")
logger.setLevel(DEBUG_IMAGE)

# Set up file logging for errors and warnings
file_handler = logging.FileHandler(LOG_DIR / 'error.log')
file_handler.setFormatter(
    logging.Formatter("[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
)
# Log errors to files
file_handler.setLevel(logging.WARNING)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)







