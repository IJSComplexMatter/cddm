"""
FFT tools. 
"""

import numpy as np
from cddm.video import VideoViewer, _FIGURES, play

class FFTViewer(VideoViewer):
    def __init__(self, title = "fft"):
        self.title = title
    
    def _prepare_image(self,im):
        im = np.fft.fftshift(np.log(1+np.abs(im)),0)
        return im/im.max()

def _window_title(name):
    i = len(_FIGURES)+1
    return "Fig.{}: {}".format(i, name)

def show_fft(video, id = 0, title = None):
    if title is None:
        title = _window_title("fft - camera {}".format(id))
    viewer = FFTViewer(title)
    
    for frames in video:
        _FIGURES[title] = (viewer, frames[id])
        yield frames
        
def _determine_cutoff_indices(shape, kisize = None, kjsize= None):
    if kisize is None:
        kisize = shape[0]
    else:    
        kisize = min(kisize, shape[0])
    if kjsize is None:
        kjsize = shape[1]
    else:
        kjsize = min(kjsize, shape[1])
    
    jstop = kjsize//2+1
    istop = kisize//2+1
    
    shape = kisize, jstop
    return shape, istop, jstop

#
#def rfft2(video, kisize = None, kjsize = None):
#    """A generator that performs rfft2 on a sequence of frames.
#    """
#    def _copy(data, vid, istop, jstop):
#        vid[:istop,:] = data[:istop,:jstop] 
#        vid[-istop:,:] = data[-istop:,:jstop] 
#        return vid
#    
#    if kisize is None and kjsize is None:
#        #just do fft, no cropping
#        for frames in video:
#            yield tuple((np.fft.rfft2(frame) for frame in frames))
#    else:
#        #do fft with cropping
#        out = None
#        for frames in video:
#            if out is None:
#                shape = frames[0].shape
#                shape, istop, jstop = _determine_cutoff_indices(shape, kisize, kjsize)
#                out = []
#                for frame in frames:
#                    data = np.fft.rfft2(frame)
#                    vid = np.empty(shape,data.dtype)
#                    out.append(_copy(data,vid,istop,jstop))
#                out = tuple(out)
#                yield out
#            else:
#                for vid, data in zip(out, frames):
#                    data = np.fft.rfft2(data)
#                    _copy(data,vid,istop,jstop) 
#                yield out
#            
#        yield out

def rfft2(video, kisize = None, kjsize = None):
    """A generator that performs rfft2 on a sequence of frames.
    """
    def _fft2(frame, shape, istop, jstop):
        data = np.fft.rfft2(frame)
        vid = np.empty(shape,data.dtype)
        vid[:istop,:] = data[:istop,:jstop] 
        vid[-istop:,:] = data[-istop:,:jstop] 
        return vid
    
    if kisize is None and kjsize is None:
        #just do fft, no cropping
        for frames in video:
            yield tuple((np.fft.rfft2(frame) for frame in frames))
    else:
        #do fft with cropping
        out = None
        for frames in video:
            if out is None:
                shape = frames[0].shape
                shape, istop, jstop = _determine_cutoff_indices(shape, kisize, kjsize)
            out = tuple((_fft2(frame,shape,istop,jstop) for frame in frames))
            yield out
   

        
if __name__ == "__main__":
    from cddm.video import random_dual_frame_video
    video = random_dual_frame_video()
    fft = rfft2(video)
    fft = show_fft(fft,0)

    for frames in play(fft, fps = 20):
        pass
