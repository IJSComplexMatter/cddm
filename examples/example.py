#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:44:51 2019

@author: andrej

This script opens a DDM video- simulated brownian motion of spherical particles
and computes autocorrelation function of the FFT of the video. Video is multiplied
by blackman window function to improve FFT data - to remove boundary effects.


If you have opencv, set USE_CV2 = True,
else, with matplotlib it will be slow, kill mpl windows after starting, or set
SHOW = False.

"""
import matplotlib.pyplot as plt
import numpy as np


SHOW = False#set this to False for fast calculation (no video preview)
USE_CV2 = False #If you have opencv installed set this to True to improve speed of preview

if USE_CV2:
    import cv2

class VideoViewer():
    """Matplotlib or opencv- based video player"""
    
    fig = None
    
    def __init__(self, title = "video"):
        self.title = title
    
    def _mpl_imshow(self,im):
        if self.fig is None:
            self.fig = plt.figure()
            self.fig.show()
            ax = self.fig.add_subplot(111)
            ax.set_title(self.title)
            self.l = ax.imshow(im)
        else:
            self.l.set_data(im)      
           
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

    def _cv_imshow(self, im):
        if self.fig is None:
            self.fig = self.title
        
        #scale from 0 to 1
        immin = im.min() 
        if immin < 0:
            im =  im - immin
        immax = im.max()
        if immax > 1:
            im = im/immax
        
        cv2.imshow(self.fig,im)
        
    def imshow(self, im):
        if USE_CV2 == True:
            self._cv_imshow(im)
            cv2.waitKey(1)
            
        else:
            self._mpl_imshow(im)
            plt.pause(0.001)  
            
    def __del__(self):
        if USE_CV2 == True:
            cv2.destroyWindow(self.fig)
        else:
            plt.close()   
            
def fit(x,a,b,f):
    return a*np.exp(-f*x)+b

def lin(x,k,n):
    return k*x+n

import scipy.optimize as opt


            
if __name__ == "__main__":
            
    video_viewer = VideoViewer(title = "video")
    fft_viewer = VideoViewer(title = "fft")
    
    vid = np.load("simple_brownian_ddm_video.npy")
    
#    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#    fps = 100
#    video_filename = "simple_brownian_ddm_video.avi"
#    out = cv2.VideoWriter(video_filename, fourcc, fps, (256,256))
#    
#    for frame in vid:
#        #gray = cv2.normalize(frame, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#        gray_3c = cv2.merge([frame, frame, frame])
#        out.write(gray_3c)
    
    bg = vid.mean() #mean pixel value
    
    ffts = []
    
    #we will multiply the video with a blackman filter, to remove boundary effects in FFT...
    #1D version:
    w = np.blackman(256)
    #2D version:
    window = np.sqrt(np.abs(np.outer(w,w))) #abs needed to remove any small negative values... 
    #window = 1.
    for i,frame in enumerate(vid):
        if i%256 == 0:
            print (i)
            
        im = frame - bg #remove mean value
        if SHOW:
            video_viewer.imshow(im)
        
        fft = np.fft.rfft2(im)[0:64,0:64] #take only first few q-values 
        
        if SHOW:
            ffta = np.abs(fft)
            fft_viewer.imshow(np.log(1+ffta/ffta.max()))
        
        ffts.append(fft.copy()) #need to copy to free memory...
        
    #make a copy of the list in one huge array... must fit into memory
    ffts = np.asarray(ffts)
    
    #dump this to disk, for later use
    np.save("simple_brownian_ddm_fft.npy", ffts)
    
    ffts = np.load("simple_brownian_ddm_fft.npy")
    
    out = []
    ks = []
    
    pixel = 6/50. #in microns
    
    popt = [1,0,1]
    k0 = 2*np.pi/256/pixel #in inverse microns
    
    for i in range(1,23):
        print(i)
    #take data at a given k values (0,22)
        signal = ffts[...,i,0]
        #compute autocorrelation
        ccorr = np.correlate(signal,signal, "full").real
        #take only positive time delays...
        ccorr= ccorr[len(ffts)-1:] 
        #perform normalization
        ccorr = ccorr / np.arange(len(ffts),0,-1) 
        
        
        ccorr = (ccorr/ccorr[0])[0:1024]
        x = np.arange(len(ccorr))/100. #in seconds
        
        popt,pcov = opt.curve_fit(fit,x,ccorr, p0 = popt)
        
        plt.semilogx(x,ccorr, "o", label = i) 
        plt.semilogx(x,fit(x,*popt), label = i) 
        
        out.append(popt[-1])
        ks.append(k0*i)

        
        #plot first few time delays
        
    plt.figure()   
    plt.plot(np.array(ks)**2,out)
    
    popt, pcov = opt.curve_fit(lin,np.array(ks)**2,out)
    
    print(popt)
    plt.show()
        
    
    
            
            