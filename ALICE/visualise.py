from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from pathlib import Path
import cv2
from numpy.typing import NDArray

def visualise_mask(image: NDArray, predictions) -> NDArray:
    """
    Return image as numpy array. 
    Use from google.colab.patches import cv2_imshow to display in colab
    """
    # img = cv2.imread(str(image_path))
    
    v = Visualizer(image[:, :, ::-1],
                   metadata=None, 
                   scale=0.7, 
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    return v.get_image()[:, :, ::-1]
    