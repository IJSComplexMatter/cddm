"""tests for video processing functions"""

import unittest
import numpy as np
from cddm.fft import rfft2
from cddm.conf import FDTYPE, set_rfft2lib
from cddm.video import asarrays, random_video, fromarrays

import numpy.fft as npfft

class TestVideo(unittest.TestCase):
    
    def setUp(self):
        video = random_video((32,42), count = 128, dtype = "uint8", max_value = 255)
        self.vid, = asarrays(video, count = 128)
        self.fft = npfft.rfft2(self.vid)
    
    def test_rfft2_numpy(self):
        set_rfft2lib("numpy")
        video = fromarrays((self.vid,))
        fft, = asarrays(rfft2(video),128)
        self.assertTrue(np.allclose(fft, self.fft))
        
        for kimax, kjmax in ((5,6), (7,7),(4,4)):
            video = fromarrays((self.vid,))
            fft, = asarrays(rfft2(video, kimax = kimax, kjmax = kjmax),128)
            self.assertTrue(np.allclose(fft[:,0:kimax+1], self.fft[:,0:kimax+1,0:kjmax+1])) 
            self.assertTrue(np.allclose(fft[:,-kimax:], self.fft[:,-kimax:,0:kjmax+1]))  

    def test_rfft2_scipy(self):
        set_rfft2lib("scipy")
        video = fromarrays((self.vid,))
        fft, = asarrays(rfft2(video),128)
        self.assertTrue(np.allclose(fft, self.fft))
        
        for kimax, kjmax in ((5,6), (7,7),(4,4)):
            video = fromarrays((self.vid,))
            fft, = asarrays(rfft2(video, kimax = kimax, kjmax = kjmax),128)
            self.assertTrue(np.allclose(fft[:,0:kimax+1], self.fft[:,0:kimax+1,0:kjmax+1])) 
            self.assertTrue(np.allclose(fft[:,-kimax:], self.fft[:,-kimax:,0:kjmax+1]))  

#    def test_rfft2_mkl(self):
#        set_rfft2lib("mkl_fft")
#        video = fromarrays((self.vid,))
#        fft, = asarrays(rfft2(video),128)
#        self.assertTrue(np.allclose(fft, self.fft))
#        
#        for kimax, kjmax in ((5,6), (7,7),(4,4)):
#            video = fromarrays((self.vid,))
#            fft, = asarrays(rfft2(video, kimax = kimax, kjmax = kjmax),128)
#            self.assertTrue(np.allclose(fft[:,0:kimax+1], self.fft[:,0:kimax+1,0:kjmax+1])) 
#            self.assertTrue(np.allclose(fft[:,-kimax:], self.fft[:,-kimax:,0:kjmax+1]))  
#

    
if __name__ == "__main__":
    unittest.main()