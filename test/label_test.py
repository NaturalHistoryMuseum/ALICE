import unittest
from pathlib import Path



from alice.models.specimen import SpecimenAngledView
from alice.utils.geometry import approx_best_fit_quadrilateral, order_points
from alice.config import TEST_DIR, logger, PROCESSING_IMAGE_DIR
from alice.models.geometric.quadrilateral import Quadrilateral
from alice.models.specimen import Specimen

class LabelTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        logger.set_specimen_id('label')
        
        # paths = [PROCESSING_IMAGE_DIR / f'011250151_additional({i}).jpg' for i in range(1,5)]
        # cls.specimen = Specimen('011250151', paths)
        
        # paths = [PROCESSING_IMAGE_DIR / f'Tri434015_additional_{i}.JPG' for i in range(1,5)]
        # cls.specimen = Specimen('Tri434015', paths)
        
        label_image  = PROCESSING_IMAGE_DIR / 'Tri434015_additional_2.JPG'        
        view = SpecimenAngledView(label_image, 1) 
        cls.image = view.image
        cls.label0 = view.labels[0]


    def test_label_crop(self):          
        # self.specimen.get_labels()
        pass
        
    def test_label_corner_correction(self):  
        pass
                  

        

        
        
    #     quad = self.hard_label0.get_quadrilateral()
    #     self.assertTrue(quad.is_wellformed_label_shape())
    #     logger.debug_image(quad.visualise(), 'label-quad-wellformed')
        

        

        
        

        
        




if __name__ == '__main__':
    unittest.main()
