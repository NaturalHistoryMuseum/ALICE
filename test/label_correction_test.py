import unittest
from pathlib import Path

from alice.models.view import AngledView
from alice.utils.geometry import approx_best_fit_quadrilateral, order_points
from alice.config import TEST_DIR, logger, PROCESSING_IMAGE_DIR
from alice.models.geometric.quadrilateral import Quadrilateral
from alice.models.specimen import Specimen
from alice.models.geometric.point import Point
from alice.log import init_log

class LabelCorrectionTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        init_log('label-correction')      
        
        label_image  = PROCESSING_IMAGE_DIR / '011250151_additional(2).JPG'        
        view = AngledView(label_image, 1)        
        cls.label011250151 = view.labels[1]  
               
        label_image  = PROCESSING_IMAGE_DIR / 'Tri434015_additional_3.JPG'        
        view = AngledView(label_image, 2)        
        cls.labelTri434015 = view.labels[1]     

    def test_invalid_corner_tri434015(self):  
        quad = self.labelTri434015._get_best_fit_polygon_quadrilateral()
        invalid_corner = self.labelTri434015.get_invalid_corner(quad)
        self.assertEqual(invalid_corner, 'c')    

    def test_edge_to_correct_tri434015(self):  
        quad = self.labelTri434015._get_best_fit_polygon_quadrilateral()
        invalid_corner = self.labelTri434015.get_invalid_corner(quad)
        edge = self.labelTri434015.get_invalid_corner_edge(invalid_corner, quad)      
        self.assertEqual(edge, 'b_c') 
        
    def test_quad_corrected_tri434015(self): 
        quad = self.labelTri434015._get_best_fit_polygon_quadrilateral()
        corrected = self.labelTri434015._get_corner_corrected_quadrilateral(quad)
        self.assertEqual(quad.vertices['a'], corrected.vertices['a'])
        self.assertNotEqual(quad.vertices['b'], corrected.vertices['b'])
        self.assertNotEqual(quad.vertices['c'], corrected.vertices['c'])
        self.assertEqual(quad.vertices['d'], corrected.vertices['d'])
        self.assertEqual(corrected.vertices['b'], Point(1235, 841))
        self.assertEqual(corrected.vertices['c'], Point(1454, 903))
                
    def test_invalid_corner_011250151(self):  
        quad = self.label011250151._get_best_fit_polygon_quadrilateral()
        invalid_corner = self.label011250151.get_invalid_corner(quad)
        self.assertEqual(invalid_corner, 'd')    

    def test_edge_to_correct_011250151(self):  
        quad = self.label011250151._get_best_fit_polygon_quadrilateral()
        invalid_corner = self.label011250151.get_invalid_corner(quad)
        edge = self.label011250151.get_invalid_corner_edge(invalid_corner, quad)    
        self.assertEqual(edge, 'd_a') 

        

        

        
        

        
        




if __name__ == '__main__':
    unittest.main()
