from alice.config import LOG_DIR, logger
import shutil

def clear_image_log(specimen_id):
    """
    Clear all files in the image log matching the specimen id 
    """
    
    log_dir = LOG_DIR / specimen_id
    if log_dir.is_dir():
        shutil.rmtree(log_dir)
        
        # log_dir.rmdir()
    
    # # Use pathlib to iterate through all files in the directory
    # for file_path in LOG_DIR.iterdir():
    #     # Check if the file is a regular file and if its name contains "a12345"
    #     if file_path.is_file() and specimen_id in file_path.name:  
    #         file_path.unlink()
            
    logger.info('Cleared image log for specimen %s', specimen_id)
        