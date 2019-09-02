#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 22:01:49 2019

@author: andrej
"""

import numpy as np
import matplotlib.pyplot as plt
import time   
from cddm.conf import CDDMConfig, CV2_INSTALLED, set_cv2, F32,F64, U16
from cddm.print_tools import print_progress, print
import numba as nb

if CV2_INSTALLED:
    import cv2
    
@nb.njit()    
def subtract_and_multiply(array, window, bg):
    tmp = array - bg
    return tmp * window

@nb.vectorize([F32(U16,F32,F32),F64(U16,F64,F64)], target = "parallel")    
def subtract_and_multiply_vec(array, window, bg):
    tmp = array - bg
    return tmp * window

def random_dual_frame_video(shape = (512,512), n = 1000):
    """"""
    for i in range(n):
        yield np.random.randn(*shape), np.random.randn(*shape)

def fromarrays(arrays):
    """Creates a multi-frame iterator from given list of arrays.
    
    Parameters
    ----------
    arrays : tuple of array-like
        A tuple of array-like objects that represent a single-camera videos
    
    Returns
    -------
    video : iterator
        A multi-frame iterator
    """
    return (frames for frames in zip(*arrays)) 

def asarrays(video, count = None):
    """Loads multi-frame video into numpy arrays. 
     
    Parameters
    ----------
    video : iterable
       A multi-frame iterator object.
    count : int, optional
       Defines how many frames are in the video. If not provided and video has
       an undefined length, it will try to load the video using np.asarray. 
       This means that data copying 
    """
    
    def _load(array, frame):
        array[...] = frame
        
    print("Writing to array...")
    
    if count is None:
        try:
            count = len(video)
        except TypeError:
            out = np.asarray(video)
            out = tuple((out[:,i] for i in range(out.shape[1])))
            return out

    print_progress(0, count)
    
    frames = next(video)
    out = tuple((np.empty(shape = (count,) + frame.shape, dtype = frame.dtype) for frame in frames))
    [_load(out[i][0],frame) for i,frame in enumerate(frames)]
    for j,frames in enumerate(video):
        print_progress(j+1, count)
        [_load(out[i][j+1],frame) for i,frame in enumerate(frames)]
        
    print_progress(count, count)
    return out

def asmemmaps(basename, video, count = None):
    """Loads multi-frame video into numpy memmaps. 
     
    Parameters
    ----------
    basename: str
       Base name for the filenames of the videos. 
    video : iterable
       A multi-frame iterator object.
    count : int, optional
       Defines how many frames are in the video. If not provided it is determined
       by len().
    """
    if count is None:
        count = len(count)
        
    def _load(array, frame):
        array[...] = frame
        
    def _empty_arrays(frames):
        out = tuple( (np.lib.format.open_memmap(basename + "_{}.npy".format(i), "w+", shape = (count,) + frame.shape, dtype = frame.dtype) 
                      for i,frame in enumerate(frames)))
        return out

    print("Writing to memmap...")
    print_progress(0, count)
    
    frames = next(video)
    out = _empty_arrays(frames)
    [_load(out[i][0],frame) for i,frame in enumerate(frames)]
    for j,frames in enumerate(video):
        print_progress(j+1, count)
        [_load(out[i][j+1],frame) for i,frame in enumerate(frames)]
    
    print_progress(count, count)   
    return out
    

class VideoViewer():
    
    fig = None
    
    def __init__(self, title = "video"):
        self.title = title
    
    def _prepare_image(self,im):
        return im
    
    def _mpl_imshow(self,im):
        if self.fig is None:
            self.fig = plt.figure()
            self.fig.show()
            ax = self.fig.add_subplot(111)
            ax.set_title(self.title)
            im = self._prepare_image(im)
            self.l = ax.imshow(im)
        else:
            im = self._prepare_image(im)
            self.l.set_data(im)      
           
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

    def _cv_imshow(self, im):
        if self.fig is None:
            self.fig = self.title
        im = self._prepare_image(im)
        
        #scale from 0 to 1
#        immin = im.min() 
#        if immin < 0:
#            im -= immin
#        immax = im.max()
#        if immax > 1:
#            im = im/immax
        
        cv2.imshow(self.fig,im)
        
    def imshow(self, im):
        if CDDMConfig.cv2 == True:
            self._cv_imshow(im)
        else:
            self._mpl_imshow(im)
            
    def __del__(self):
        if CDDMConfig.cv2 == True:
            cv2.destroyWindow(self.fig)
        else:
            plt.close()       
    
def _pause():
    if CDDMConfig.cv2 == False:
        plt.pause(0.01)  
    else:
        cv2.waitKey(1)
             
def play(video, fps = 100):
    t0 = None
    update = True
    for i, frames in enumerate(video):
        if t0 is None:
            t0 = time.time()
        if update == True:
            for key in list(_FIGURES.keys()):
                (viewer, im) = _FIGURES.pop(key)
                viewer.imshow(im)
            _pause()
            
        yield frames
        
        if time.time()-t0 < i/fps:
            update = True
        else:
            update = False  
    _FIGURES.clear()
        
_FIGURES = {}

  
def apply_window(video, window, inplace = False):
    for frames in video:
        if inplace == True:
            yield tuple((np.multiply(frame, w, frame) for w, frame in zip(window,frames)))
        else:
            yield tuple((frame*w for w, frame in zip(window,frames)))
            
            
def show_video(video, id = 0):
    title = "video - camera {}".format(id)
    viewer = VideoViewer(title)
    
    for frames in video:
        _FIGURES[title] = (viewer, frames[id])
        yield frames
        
     
def show_diff(video):
    title = "video - difference"
    viewer = VideoViewer(title)
    
    for frames in video:
        if not title in _FIGURES:
            m = 2* max(frames[0].max(),frames[1].max())
            im = frames[0]/m - frames[1]/m + 0.5
            _FIGURES[title] = (viewer, im)
        yield frames    
        

if __name__ == '__main__':

    video = random_dual_frame_video()
    video = show_video(video)
    video = show_diff(video)
    v1,v2 = load_array(play(video, fps = 20),1000)

    #for frames in play(video, fps = 20):
    #    pass
    
