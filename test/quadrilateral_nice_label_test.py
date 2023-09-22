import unittest
from pathlib import Path



from alice.models.specimen import SpecimenAngledView
from alice.utils.geometry import approx_best_fit_quadrilateral, order_points
from alice.config import TEST_DIR, logger
from alice.models.geometric.quadrilateral import Quadrilateral


class QuadrilateralTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        logger.set_specimen_id('label')
        label_image  = TEST_DIR / 'images/Tri434012_additional_4.JPG'
        view = SpecimenAngledView(label_image, 1)   
        
        cls.image = view.image
        cls.label0 = view.labels[0]
        cls.label1 = view.labels[1]        

    def test_assert_is_wellformed0(self):  
        quad = Quadrilateral(self.label0.vertices, self.image)
        self.assertTrue(quad.is_wellformed_label_shape())
        
    def test_assert_is_wellformed1(self):  
        quad = Quadrilateral(self.label1.vertices, self.image)
        self.assertTrue(quad.is_wellformed_label_shape())        
                    


if __name__ == '__main__':
    unittest.main()
