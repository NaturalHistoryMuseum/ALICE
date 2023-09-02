from pathlib import Path
import logging
from enum import Enum
import os

DEBUG = os.getenv('DEBUG') or 1

# Maximum number of labels to look at per image.
MAX_NUMBER_OF_LABELS = 6

###### Paths #######

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = Path(ROOT_DIR / 'data')

ASSETS_DIR = Path(DATA_DIR / 'assets')
MODEL_DIR = Path(DATA_DIR / 'models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_DIR = Path(DATA_DIR / 'evaluation')
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(ROOT_DIR / '.cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path(ROOT_DIR / '.log')
LOG_DIR.mkdir(parents=True, exist_ok=True)

VISUALISATION_DIR = Path(DATA_DIR / 'visualisations')
VISUALISATION_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATA_DIR = Path(DATA_DIR / 'output')
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATASET_PATH = DATA_DIR / 'label' / 'train'
VAL_DATASET_PATH = DATA_DIR / 'label' / 'val'

IMAGE_BASE_WIDTH = 2048
PROCESSING_INPUT_DIR = Path(DATA_DIR / 'input')
PROCESSING_INTERMEDIATE_DIR = Path(DATA_DIR / 'intermediate')
PROCESSING_MASK_DIR = Path(PROCESSING_INTERMEDIATE_DIR / 'masks')
PROCESSING_IMAGE_DIR = Path(PROCESSING_INTERMEDIATE_DIR / 'images')
PROCESSING_OUTPUT_DIR = Path(DATA_DIR / 'output')
PROCESSING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSING_NUM_CAMERA_VIEWS = 4 

if IMAGE_BASE_WIDTH:
    resized_dir = f'{IMAGE_BASE_WIDTH}x'
    PROCESSING_MASK_DIR /= resized_dir
    PROCESSING_IMAGE_DIR /= resized_dir

PROCESSING_MASK_DIR.mkdir(parents=True, exist_ok=True)
PROCESSING_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


    

    

MODEL_DIR = Path(DATA_DIR / 'models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)



###### Train config #######

NUM_EPOCHS = 1





# # Directory to save logs and model checkpoints, if not provided
# # through the command line argument --logs
# DEFAULT_LOGS_DIR = DATA_DIR / "logs"

###### Logging #######

# Set up logging - inherit from luigi so we use the same interface
logger = logging.getLogger('luigi-interface')

# Capture all log levels, but handlers below set their own levels
logger.setLevel(logging.DEBUG)

# Set up file logging for errors and warnings
file_handler = logging.FileHandler(LOG_DIR / 'error.log')
file_handler.setFormatter(
    logging.Formatter("[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
)
# Log errors to files
file_handler.setLevel(logging.WARNING)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)







