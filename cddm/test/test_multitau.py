import unittest
import numpy as np
import cddm.multitau as multitau
from cddm.core import stats
from cddm.conf import FDTYPE, CDTYPE
from cddm.video import fromarrays

class TestMulti(unittest.TestCase):
    
    def setUp(self):
        self.test_data1 = np.random.randn(64,19,8) + np.random.randn(64,19,8)*1j
        self.test_data2 = np.random.randn(64,19,8) + np.random.randn(64,19,8)*1j
        self. test_data1 = np.array(self.test_data1, CDTYPE)
        self.test_data2 = np.array(self.test_data2, CDTYPE)
    
    def test_equivalence_norm_1(self):
        norm = 1
        bg, var = stats(self.test_data1)
        data = multitau.acorr_multi(self.test_data1, level_size = 16, norm = norm)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out0 = multitau.log_merge(*data)
        data = multitau.ccorr_multi(self.test_data1,self.test_data1, level_size = 16, norm = norm)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out = multitau.log_merge(*data)
        self.assertTrue(np.allclose(out0,out))
        
        data, bg, var = multitau.iacorr_multi(fromarrays((self.test_data1,)),count = 64, level_size = 16,  norm = norm)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out = multitau.log_merge(*data)
        self.assertTrue(np.allclose(out0,out))
 
    def test_equivalence_norm_2(self):
        norm = 2
        bg, var = stats(self.test_data1)
        data= multitau.acorr_multi(self.test_data1, level_size = 16, norm = norm)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out0 = multitau.log_merge(*data)
        data = multitau.ccorr_multi(self.test_data1,self.test_data1, level_size = 16, norm = norm)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out = multitau.log_merge(*data)
        self.assertTrue(np.allclose(out0,out))
        
        data,bg,var = multitau.iacorr_multi(fromarrays((self.test_data1,)),count = 64, level_size = 16,  norm = norm)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out = multitau.log_merge(*data)
        self.assertTrue(np.allclose(out0,out))

    def test_equivalence_norm_3(self):
        norm = 3
        bg, var = stats(self.test_data1)
        data= multitau.acorr_multi(self.test_data1, level_size = 16, norm = norm)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out0 = multitau.log_merge(*data)
        data = multitau.ccorr_multi(self.test_data1,self.test_data1, level_size = 16, norm = norm)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out = multitau.log_merge(*data)
        self.assertTrue(np.allclose(out0,out))
        
        data,bg,var = multitau.iacorr_multi(fromarrays((self.test_data1,)),count = 64, level_size = 16,  norm = norm)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out = multitau.log_merge(*data)
        self.assertTrue(np.allclose(out0,out))
        
    def test_equivalence_diff_2(self):
        norm = 2
        bg, var = stats(self.test_data1)
        data= multitau.acorr_multi(self.test_data1, level_size = 16, norm = norm, method = "corr", binning = 0)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out0 = multitau.log_merge(*data)
        data = multitau.ccorr_multi(self.test_data1,self.test_data1, level_size = 16, norm = norm, method = "diff", binning = 0)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out = multitau.log_merge(*data)
        self.assertTrue(np.allclose(out0,out))
        
        data,bg,var = multitau.iacorr_multi(fromarrays((self.test_data1,)),count = 64, level_size = 16,  norm = norm, method = "diff", binning = 0)
        data = multitau.normalize_multi(data,bg,var, norm = norm)
        x_, out = multitau.log_merge(*data)
        self.assertTrue(np.allclose(out0,out))
        
        
if __name__ == "__main__":
    unittest.main()
    
                        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    