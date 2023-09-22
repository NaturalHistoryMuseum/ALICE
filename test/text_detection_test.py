import unittest
from pathlib import Path
import cv2


from alice.models.text import TextDetection
from alice.utils.geometry import approx_best_fit_quadrilateral, order_points
from alice.config import TEST_DIR, logger, PROCESSING_IMAGE_DIR
from alice.models.geometric.quadrilateral import Quadrilateral


class TextDetectionTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        files011244568 = ['011244568-label-0-0.jpg', '011244568-label-0-1.jpg', '011244568-label-0-3.jpg']
        files011244568 = ['011244568-label-1-0.jpg', '011244568-label-1-2.jpg', '011244568-label-1-3.jpg']
        cls.images011244568 = [cv2.imread(str(TEST_DIR / 'images' / f)) for f in files011244568]

    # def test_detection_011244568_0(self):    
    #     text = TextDetection(self.images011244568[0])
    #     self.assertEqual(len(text), 4)

    # def test_detection_011244568_1(self):    
    #     text = TextDetection(self.images011244568[1])    
    #     self.assertEqual(len(text), 4)    
        
    # def test_detection_011244568_recalculate_cluster(self):    
    #     text = TextDetection(self.images011244568[2])    
    #     self.assertEqual(len(text), 4)      
    #     text.recalculate_clusters(3)
    #     self.assertEqual(len(text), 3) 
                    
    def test_detection_011244568_1_0(self):    
        text = TextDetection(self.images011244568[0])
        self.assertEqual(len(text), 3)

    def test_detection_011244568_1_1(self):    
        text = TextDetection(self.images011244568[1])
        self.assertEqual(len(text), 3)    
        
    def test_detection_011244568_1_2(self):    
        text = TextDetection(self.images011244568[2])    
        self.assertEqual(len(text), 3)           
        



if __name__ == '__main__':
    unittest.main()