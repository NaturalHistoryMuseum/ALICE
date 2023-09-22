import unittest
from pathlib import Path


from alice.config import TEST_IMAGE_DIR, logger
from alice.models.labels.specimen import Specimen
from alice.log import init_log

class LabelQuartetTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        init_log('label-quartet')                  
        paths = [TEST_IMAGE_DIR / f'Tri434015_additional_{i}.jpg' for i in range(1,5)]
        cls.tri434015_specimen = Specimen('Tri434015', paths)
        

    def test_tri434015_num_quartets(self):  
        # 2 labels on specimen Tri434015
        quartets = self.tri434015_specimen.get_label_quartets()
        self.assertEqual(len(quartets), 2)
        

 

if __name__ == '__main__':
    unittest.main()
