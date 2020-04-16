"""tests for video processing functions"""

import unittest
import numpy as np
from cddm.video import subtract, multiply, normalize_video, random_video, asmemmaps,\
     asarrays, fromarrays, load, crop, show_video, show_fft, show_diff, play, add
from cddm.conf import FDTYPE
from cddm.window import blackman


video = random_video((32,8), count = 128, dtype = "uint8", max_value = 255)
vid, = asarrays(video, count = 128)

bg = vid.mean(0)
window = blackman((32,8))

vid_subtract = vid - bg[None,...]
vid_multiply = vid * window[None,...]
vid_add = vid + bg[None,...]
vid_normalize = vid / (vid.mean((1,2))[:,None,None])
vid_multiple = vid_subtract * window[None,...]
vid_multiple = vid_multiple / (vid_multiple.mean((1,2))[:,None,None])

class TestVideo(unittest.TestCase):
    
    def setUp(self):
        pass  
    
    def test_memmaps(self):
        video = fromarrays((vid,))
        with self.assertRaises(ValueError):
            video = asmemmaps("deleteme", video)
        video = asmemmaps("deleteme", video,128)   
    
    def test_subtract(self):
        video = fromarrays((vid,))
        with self.assertRaises(ValueError):
            list(subtract(video, ((bg,bg),)*128))
        video = fromarrays((vid,))
        out = subtract(video, ((bg,),)*128)    
        for frames, true_frame in zip(out, vid_subtract):
            self.assertTrue(np.allclose(frames[0],true_frame))
            
    def test_crop(self):
        video = fromarrays((vid,))
        with self.assertRaises(ValueError):
            list(crop(video, roi = ((0,4),0,2)))
        out, = asarrays(crop(video, roi = ((0,2),(0,2))),128)
        self.assertTrue(np.allclose(out[0], vid[0,0:2,0:2]))
        
    
    def test_load(self):
        video = fromarrays((vid,))
        with self.assertRaises(ValueError):
            video = load(video)
        video = load(video,128)
        video = load(video)

    def test_add(self):
        video = fromarrays((vid,))
        with self.assertRaises(ValueError):
            list(add(video, ((bg,bg),)*128))
        video = fromarrays((vid,))
        out = add(video, ((bg,),)*128)
        for frames, true_frame in zip(out, vid_add):
            self.assertTrue(np.allclose(frames[0],true_frame))
        
    def test_multiply(self):
        video = fromarrays((vid,))
        with self.assertRaises(ValueError):
            list(multiply(video, ((window,window),)*128))
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
        video = add(video, ((bg,),)*128, inplace = True)
        video = subtract(video, ((bg,),)*128, inplace = True)
        out = normalize_video(video, inplace = True)
        
        for frames, true_frame in zip(out, vid_multiple):
            self.assertTrue(np.allclose(frames[0],true_frame))
            
    def test_show(self):
        video = fromarrays((vid,vid))
        video = show_video(video)
        video = show_fft(video)
        video = show_diff(video)
        video = play(video, fps = 100)
        video = load(video, 128)
        
    
if __name__ == "__main__":
    unittest.main()