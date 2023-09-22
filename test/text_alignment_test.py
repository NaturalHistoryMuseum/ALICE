import unittest
from pathlib import Path
import cv2


from alice.models.text import TextDetection
from alice.utils.geometry import approx_best_fit_quadrilateral, order_points
from alice.config import TEST_DIR, logger, PROCESSING_IMAGE_DIR
from alice.models.geometric.quadrilateral import Quadrilateral
from alice.models.text.alignment import TextAlignment
from alice.log import init_log

class TextAlignmentTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        init_log('test-alignment')
        
        files011244568_0 = ['011244568-label-0-0.jpg', '011244568-label-0-1.jpg', '011244568-label-0-3.jpg']
        files011244568_1 = ['011244568-label-1-0.jpg', '011244568-label-1-2.jpg', '011244568-label-1-3.jpg']
        images011244568_0 = [cv2.imread(str(TEST_DIR / 'images' / f)) for f in files011244568_0]
        images011244568_1 = [cv2.imread(str(TEST_DIR / 'images' / f)) for f in files011244568_1]

        # All of the labels for 011244568 at label level 0
        # cls.detectors_011244568_0 = [TextDetection(image) for image in images011244568_0]
        cls.detectors_011244568_1 = [TextDetection(image) for image in images011244568_1]


    def test_alignment_011244568_1_line0(self):   
        init_log('test-alignment-011244568-1-0')
        line_index = 0     
        lines = [detector.text_lines[line_index] for detector in self.detectors_011244568_1]
        alignment = TextAlignment(lines)
        
    def test_alignment_011244568_1_line1(self):   
        init_log('test-alignment-011244568-1-1')
        line_index = 1
        lines = [detector.text_lines[line_index] for detector in self.detectors_011244568_1]
        alignment = TextAlignment(lines)
        
    def test_alignment_011244568_1_line2(self):   
        init_log('test-alignment-011244568-1-2')
        line_index = 2     
        lines = [detector.text_lines[line_index] for detector in self.detectors_011244568_1]
        alignment = TextAlignment(lines)
                                                     
    # def test_alignment_011244568_0_line0(self):   
    #     init_log('test-alignment-011244568-0-0')
    #     line_index = 0     
    #     lines = [detector.text_lines[line_index] for detector in self.detectors_011244568_0]
    #     alignment = TextAlignment(lines)
    #     self._assert_dimensions(alignment.composite, 416, 41)
        
    # def test_alignment_011244568_0_line1(self):   
    #     init_log('test-alignment-011244568-0-1')
    #     line_index = 1     
    #     lines = [detector.text_lines[line_index] for detector in self.detectors_011244568_0]
    #     alignment = TextAlignment(lines)
    #     self._assert_dimensions(alignment.composite, 416, 41)

    # def test_alignment_011244568_0_line2(self):   
    #     init_log('test-alignment-011244568-0-2')
    #     line_index = 2
    #     lines = [detector.text_lines[line_index] for detector in self.detectors_011244568_0]
    #     alignment = TextAlignment(lines)
    #     self._assert_dimensions(alignment.composite, 416, 61)
        
    # def test_alignment_011244568_0_line3(self):   
    #     init_log('test-alignment-011244568-0-3')
    #     line_index = 3     
    #     lines = [detector.text_lines[line_index] for detector in self.detectors_011244568_0]
    #     alignment = TextAlignment(lines)
    #     self._assert_dimensions(alignment.composite, 416, 75)
             
    def _assert_dimensions(self, image, width, height): 
        h, w = image.shape[:2]
        self.assertEqual(w, width)
        self.assertEqual(h, height)         
        

        
        
        
       
        



if __name__ == '__main__':
    unittest.main()