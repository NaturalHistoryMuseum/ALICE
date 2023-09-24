from pathlib import Path
import logging
from enum import Enum
import os
import cv2
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

DEBUG = os.getenv('DEBUG', 1)

# Maximum number of labels to look at per image.
# FIXME: I'm not foing anything with MAX_NUMBER_OF_LABELS
MAX_NUMBER_OF_LABELS = 6

IMAGE_BASE_WIDTH = 2048
NUM_CAMERA_VIEWS = 4 

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

LOG_DIR = DATA_DIR / 'log'
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATASET_PATH = DATA_DIR / 'label' / 'train'
VAL_DATASET_PATH = DATA_DIR / 'label' / 'val'

INPUT_DIR = Path(os.getenv("INPUT_DIR", Path(DATA_DIR / 'input')))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(DATA_DIR / 'output')))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESIZED_IMAGE_DIR = Path(DATA_DIR / 'resized')
RESIZED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(DATA_DIR / 'models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)


###### Train config #######

NUM_EPOCHS = 1

###### Logging #######

# Define a custom log level
LOG_LEVEL_DEBUG_IMAGE = 15

# Add the custom log level to the logging module
logging.addLevelName(LOG_LEVEL_DEBUG_IMAGE, 'LOG_LEVEL_DEBUG_IMAGE')

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
            
        if self.isEnabledFor(LOG_LEVEL_DEBUG_IMAGE):
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

if DEBUG:
    logger.setLevel(LOG_LEVEL_DEBUG_IMAGE)
else:
    logger.setLevel(logging.INFO)
    
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







