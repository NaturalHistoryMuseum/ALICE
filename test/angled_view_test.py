import unittest
from pathlib import Path



from alice.models.view import AngledView
from alice.utils.geometry import approx_best_fit_quadrilateral, order_points
from alice.config import TEST_DIR, logger, PROCESSING_IMAGE_DIR
from alice.models.geometric.quadrilateral import Quadrilateral
from alice.models.specimen import Specimen

class AngledViewTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        logger.set_specimen_id('angled-view')        
        label_image  = PROCESSING_IMAGE_DIR / 'Tri434015_additional_2.JPG'        
        cls.view = AngledView(label_image, 1) 
                
    def test_num_labels(self):  
        self.assertEqual(len(self.view.labels), 2) 

    def test_num_masks_equals_labels(self):  
        self.assertEqual(len(self.view.labels), len(self.view.label_masks)) 

        

        

        
        

        
        




if __name__ == '__main__':
    unittest.main()
