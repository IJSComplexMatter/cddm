"""tests for video processing functions"""

import unittest
import numpy as np
from cddm.fft import rfft2, normalize_fft, _ifft, _fft
from cddm.conf import FDTYPE, set_rfft2lib, set_fftlib
from cddm.video import asarrays, random_video, fromarrays

import numpy.fft as npfft

class TestVideo(unittest.TestCase):
    
    def setUp(self):
        video = random_video((31,32), count = 128, dtype = "uint8", max_value = 255)
        self.vid, = asarrays(video, count = 128)
        self.fft = npfft.rfft2(self.vid)
        self.fft_norm = self.fft/(self.fft[:,0,0])[:,None,None]
    
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

        video = fromarrays((self.vid,))
        fft, = asarrays(rfft2(video, kimax = None, kjmax = 6),128)
        self.assertTrue(np.allclose(fft, self.fft[:,:,0:7])) 
        
        video = fromarrays((self.vid,))
        fft, = asarrays(rfft2(video, kimax = 6),128)
        self.assertTrue(np.allclose(fft[:,0:7,:], self.fft[:,0:7,:])) 
        
        with self.assertRaises(ValueError):
            video = fromarrays((self.vid,))
            fft, = asarrays(rfft2(video, kimax = 16),128)
        with self.assertRaises(ValueError):
            video = fromarrays((self.vid,))
            fft, = asarrays(rfft2(video, kjmax = 17),128)

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

    def test_normalize(self):
        video = fromarrays((self.vid,))
        fft = rfft2(video)
        fft, = asarrays(normalize_fft(fft),128)
        self.assertTrue(np.allclose(fft, self.fft_norm))
        video = fromarrays((self.vid,))
        fft = rfft2(video)
        fft, = asarrays(normalize_fft(fft, inplace = True),128)
        self.assertTrue(np.allclose(fft, self.fft_norm))
        
    def test_fft(self):
        for libname in ("numpy","scipy"):
            set_fftlib(libname)
            a = np.random.randn(4)
            fft = np.fft.fft(a)
            out = _fft(a)
            self.assertTrue(np.allclose(fft, out))

    def test_ifft(self):
        for libname in ("numpy","scipy"):
            set_fftlib(libname)
            a = np.random.randn(4)
            fft = np.fft.fft(a)
            a = np.fft.ifft(fft)
            out = _ifft(fft)
            self.assertTrue(np.allclose(a, out))
        

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