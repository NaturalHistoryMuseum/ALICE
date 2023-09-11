from alice.config import LOG_DIR, logger
import shutil

def clear_image_log(specimen_id):
    """
    Clear all files in the image log matching the specimen id 
    """
    
    log_dir = LOG_DIR / specimen_id
    if log_dir.is_dir():
        shutil.rmtree(log_dir)

    logger.info('Cleared image log for specimen %s', specimen_id)

def init_log(specimen_id):
    clear_image_log(specimen_id)
    logger.set_specimen_id(specimen_id)
    