"""tests for video processing functions"""

import unittest
import numpy as np
from cddm.fft import rfft2
from cddm.conf import FDTYPE
from cddm.video import asarrays, random_video, fromarrays

import numpy.fft as npfft

class TestVideo(unittest.TestCase):
    
    def setUp(self):
        video = random_video((32,42), count = 128, dtype = "uint8", max_value = 255)
        self.vid, = asarrays(video, count = 128)
        self.fft = npfft.rfft2(self.vid)
    
    def test_rfft2(self):
        video = fromarrays((self.vid,))
        fft, = asarrays(rfft2(video),128)
        self.assertTrue(np.allclose(fft, self.fft))
        video = fromarrays((self.vid,))
        fft, = asarrays(rfft2(video, kimax = 5, kjmax =6),128)
        self.assertTrue(np.allclose(fft[:,0:6,0:7], self.fft[:,0:6,0:7])) 
        self.assertTrue(np.allclose(fft[:,-5:,0:7], self.fft[:,-5:,0:7]))  

    
if __name__ == "__main__":
    unittest.main()