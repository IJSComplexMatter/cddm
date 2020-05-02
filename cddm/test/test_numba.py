"""tests numba functions"""

import unittest
import numpy as np
from cddm._core_nb import median, decreasing, increasing, _median_slow


class TestNumba(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_median(self):
        out = median([2.,3.,0.])
        self.assertTrue(np.allclose([2.,2.,0.], out))
        out = median([2.,3.,0.,4.,5])
        self.assertTrue(np.allclose([2.,2.,3.,4.,5.], out))
        a = np.random.rand(100,2,4)
        b = a.copy()
        self.assertTrue(np.allclose(median(b,b), _median_slow(a)))
        
        
    def test_decreasing(self):
        out = decreasing([2.,1.,2.])
        self.assertTrue(np.allclose([2.,1.,1.], out))
        out = decreasing([2.,3.,0.,4.,5])
        self.assertTrue(np.allclose([2.,2.,0.,0.,0.], out))   
        
    def test_increasing(self):
        out = increasing([2.,1.,3.])
        self.assertTrue(np.allclose([2.,2.,3.], out))
        out = increasing([2.,3.,0.,4.,5])
        self.assertTrue(np.allclose([2.,3.,3.,4.,5.], out)) 
        
if __name__ == "__main__":
    unittest.main()