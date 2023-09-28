from abc import ABCMeta, abstractmethod
from matplotlib import pyplot as plt

from alice.config import LOG_DIR


class Base(metaclass=ABCMeta):

    def __init__(self, image):
        self.image = image
        self.image_height, self.image_width = image.shape[:2]
        
    def visualise(self, image=None):
        image = self.image.copy() if image is None else image
        return self._visualise(image)

    @abstractmethod
    def _visualise(self, image):
        return None

    def display(self):
        image = self.visualise()
        plt.imshow(image), plt.show() 
            
        
        
        
