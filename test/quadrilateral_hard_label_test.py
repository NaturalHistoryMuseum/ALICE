import unittest
from pathlib import Path



from alice.models.view import AngledView
from alice.utils.geometry import approx_best_fit_quadrilateral, order_points
from alice.config import TEST_DIR, logger
from alice.models.geometric.quadrilateral import Quadrilateral


class QuadrilateralTest(unittest.TestCase):
     
    @classmethod
    def setUpClass(cls):
        logger.set_specimen_id('label')
        label_image  = TEST_DIR / 'images/011250151_additional(2).JPG'
        label_image  = TEST_DIR / 'images/011250151_additional(1).JPG'
        
        view = AngledView(label_image, 1)   
        
        cls.image = view.image
        cls.label0 = view.labels[0]
        cls.label1 = view.labels[1]    
                      
    def test_assert_not_wellformed(self):  
        quad = self.label1._get_best_fit_polygon_quadrilateral()
        self.assertFalse(quad.is_wellformed_label_shape())

    # def test_assert_has_invalid_corner1(self):  
    #     quad = Quadrilateral(self.label1.vertices, self.image)        
    #     invalid_corners = quad.get_invalid_corners()
    #     self.assertEqual(len(invalid_corners), 1)        
    #     logger.debug_image(quad.visualise(), f'invalid-corner-{invalid_corners[0]}')

    # def test_assert_has_valid_corner0(self):  
    #     quad = Quadrilateral(self.label0.vertices, self.image)        
    #     invalid_corners = quad.get_invalid_corners()
    #     self.assertEqual(len(invalid_corners), 0)        
    #     logger.debug_image(quad.visualise(), f'valid-corner')
   
    # def test_quad_corner_corrected(self):          
    #     quad = Quadrilateral(self.label1.vertices, self.label1.image)
    #     corrected_quad = self.label1.correct_invalid_corner(quad)
    #     logger.debug_image(corrected_quad.visualise(), 'corrected-corner-quad')
    #     self.assertTrue(corrected_quad.is_wellformed_label_shape())           
        
    # def test_quad_nearest_corner_guestimation(self):          
    #     quad = Quadrilateral(self.label0.vertices, self.label0.image)
    #     approx_quad = self.label0.approx_quadrilateral_from_closest_edges(quad)
    #     logger.debug_image(approx_quad.visualise(), 'corrected-nearest-corner-0')

        


if __name__ == '__main__':
    unittest.main()
