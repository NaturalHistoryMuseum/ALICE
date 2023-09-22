import unittest
from pathlib import Path
from copy import deepcopy

from alice.models.labels.view import AngledView
from alice.utils.geometry import approx_best_fit_quadrilateral, order_points
from alice.config import TEST_DIR, logger, PROCESSING_IMAGE_DIR
from alice.models.geometric.quadrilateral import Quadrilateral
from alice.models.labels.specimen import Specimen
from alice.models.geometric.point import Point
from alice.models.labels import Label, InvalidLabel, LabelValid
from alice.models.labels.crop import CropLabels
from alice.log import init_log

class LabelCropTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        init_log('test-label-crop')           
        cls.view1 = AngledView(PROCESSING_IMAGE_DIR / 'Tri434015_additional_1.JPG', 1)
        cls.view2 = AngledView(PROCESSING_IMAGE_DIR / 'Tri434015_additional_2.JPG', 2) 
        # cls.view3 = AngledView(PROCESSING_IMAGE_DIR / 'Tri434015_additional_3.JPG', 3) 

    def test_single_label_crop(self):  
        label = deepcopy(self.view1.labels[0])
        cropped = CropLabels([label]).crop()
        logger.debug_image(cropped[0], f'cropped-tri434015-single')
 
    def test_double_label_crop(self):  
        label1 = deepcopy(self.view1.labels[0])
        label2 = deepcopy(self.view2.labels[0])
        cropped = CropLabels([label1, label2]).crop()
        print(cropped)
        for i, crop in cropped.items():   
            print(crop)     
            print(crop.shape)
            logger.debug_image(crop, f'cropped-tri434015-double-{i}')
 
    # def test_triple_label_crop(self):  
    #     label1 = self.view1.labels[0]
    #     label2 = self.view2.labels[0]
    #     label3 = self.view3.labels[0]
    #     cropped = CropLabels([label1, label2, label3]).crop()
    #     logger.debug_image(cropped[0], f'cropped-tri434015-triple-0')
    #     logger.debug_image(cropped[1], f'cropped-tri434015-triple-1')
    #     logger.debug_image(cropped[2], f'cropped-tri434015-triple-2')
 
if __name__ == '__main__':
    unittest.main()






        

        

        
        

        
        




if __name__ == '__main__':
    unittest.main()
