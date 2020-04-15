"""tests for video processing functions"""

import unittest
import numpy as np
from cddm.video import subtract, multiply, normalize_video, random_video, asarrays, fromarrays
from cddm.conf import FDTYPE
from cddm.window import blackman


video = random_video((32,8), count = 128, dtype = "uint8", max_value = 255)
vid, = asarrays(video, count = 128)

bg = vid.mean(0)
window = blackman((32,8))

vid_subtract = vid - bg[None,...]
vid_multiply = vid * window[None,...]
vid_normalize = vid / (vid.mean((1,2))[:,None,None])
vid_multiple = vid_subtract * window[None,...]
vid_multiple = vid_multiple / (vid_multiple.mean((1,2))[:,None,None])

class TestVideo(unittest.TestCase):
    
    def setUp(self):
        pass  
    
    def test_subtract(self):
        video = fromarrays((vid,))
        out = subtract(video, ((bg,),)*128)
        for frames, true_frame in zip(out, vid_subtract):
            self.assertTrue(np.allclose(frames[0],true_frame))

    def test_multiply(self):
        video = fromarrays((vid,))
        out = multiply(video, ((window,),)*128)
        for frames, true_frame in zip(out, vid_multiply):
            self.assertTrue(np.allclose(frames[0],true_frame))
 
    def test_normalize(self):
        video = fromarrays((vid,))
        out = normalize_video(video)
        for frames, true_frame in zip(out, vid_normalize):
            self.assertTrue(np.allclose(frames[0],true_frame))
            
    def test_multiple(self):
        video = fromarrays((vid,))
        video = subtract(video, ((bg,),)*128, dtype = FDTYPE)
        video = multiply(video, ((window,),)*128, inplace = True)
        out = normalize_video(video, inplace = True)
        
        for frames, true_frame in zip(out, vid_multiple):
            self.assertTrue(np.allclose(frames[0],true_frame))        
        
       
if __name__ == "__main__":
    unittest.main()