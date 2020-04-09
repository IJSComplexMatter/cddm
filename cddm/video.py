"""
Video processing tools. 

You can use this helper functions to perform normalization, background subtraction 
and windowing on multi-frame data. 

There are also function for real-time display of videos for real-time analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import time   
from cddm.conf import CDDMConfig, CV2_INSTALLED,  FDTYPE
from cddm.print_tools import print_progress, print1, print_frame_rate
from queue import Queue

if CV2_INSTALLED:
    import cv2
    
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
        Defines how many frames are in the video. If not provided it will calculate
        length of the video based on the length of the iterable. If that is not
        possible ValueError is raised
       
    Returns
    -------
    out : tuple of arrays
        A tuple of array(s) representing video(s)
    """
    
    t0 = time.time()
    
    def _load(array, frame):
        array[...] = frame
        
    print1("Loading array...")
    
    if count is None:
        try:
            count = len(video)
        except TypeError:
            raise ValueError("You must provide count")

    print_progress(0, count)
    
    video = iter(video)
    
    frames = next(video)
    out = tuple((np.empty(shape = (count,) + frame.shape, dtype = frame.dtype) for frame in frames))
    [_load(out[i][0],frame) for i,frame in enumerate(frames)]
    for j,frames in enumerate(video):
        print_progress(j+1, count)
        [_load(out[i][j+1],frame) for i,frame in enumerate(frames)]
        
    print_progress(count, count)
    print_frame_rate(count,t0)
    return out

def asmemmaps(basename, video, count = None):
    """Loads multi-frame video into numpy memmaps. 
    
    Actual data is written to numpy files with the provide basename and
    subscripted by source identifier (index), e.g. "basename_0.npy" and "basename_1.npy"
    in case of dual-frame video source.
     
    Parameters
    ----------
    basename: str
       Base name for the filenames of the videos. 
    video : iterable
       A multi-frame iterator object.
    count : int, optional
       Defines how many multi-frames are in the video. If not provided it is determined
       by len().
       
    Returns
    -------
    out : tuple of arrays
        A tuple of memmapped array(s) representing video(s)
    """
    
    if count is None:
        count = len(count)
        
    def _load(array, frame):
        array[...] = frame
        
    def _empty_arrays(frames):
        out = tuple( (np.lib.format.open_memmap(basename + "_{}.npy".format(i), "w+", shape = (count,) + frame.shape, dtype = frame.dtype) 
                      for i,frame in enumerate(frames)))
        return out

    print1("Writing to memmap...")
    print_progress(0, count)
    
    frames = next(video)
    out = _empty_arrays(frames)
    [_load(out[i][0],frame) for i,frame in enumerate(frames)]
    for j,frames in enumerate(video):
        print_progress(j+1, count)
        [_load(out[i][j+1],frame) for i,frame in enumerate(frames)]
    
    print_progress(count, count)   
    return out
 

def crop(video, roi = (slice(None), slice(None))):
    """Crops each frame in video 
    
    Parameters
    ----------
    video : iterable
        Input multi-frame iterable object. Each element of the iterable is a tuple
        of ndarrays (frames)
    roi : tuple 
        A tuple of two slice objects for slicing in first axis (height) and the
        second axis (width). You can also provide a tuple arguments tuple
        that are past to the slice builtin function. 
        
    Returns
    -------
    video : iterator
        A multi-frame iterator
        
    Examples
    --------
    One option is to provide roi with indices. To crop frames like frame[10:100,20:120]
    
    >>> video = random_video(count = 100)
    >>> video = crop(video, roi = ((10,100),(20,120)))
    
    Or you can use slice objects to perform crop
    
    >>> video = crop(video, roi = (slice(10,100),slice(20,120)))
    """
    try:    
        hslice, wslice = roi
        if not isinstance(hslice, slice) :
            hslice = slice(*hslice)
        if not isinstance(wslice, slice) :
            wslice = slice(*wslice)   
    except:
        raise ValueError("Invalid roi")
    for frames in video:
        yield tuple((frame[hslice,wslice] for frame in frames))

def subtract(x, y, inplace = False, dtype = None):
    """Subtracts each of the frames in multi-frame video with a given arrays.
    
    Parameters
    ----------
    x, y : iterable
        Input multi-frame iterable object. Each element of the iterable is a tuple
        of ndarrays (frames)
    inplace : bool, optional
        Whether tranformation is performed inplace or not. 
    dtype : numpy dtype
        If specifed, determines output dtype. Only valid if inplace == False.
                
    Returns
    -------
    video : iterator
        A multi-frame iterator
    """

    for frames, arrays in zip(x,y):
        if len(frames) != len(arrays):
            raise ValueError("Number of frames in x and y do not match")
        if inplace == True:
            yield tuple((np.subtract(frame, w, frame) for w, frame in zip(arrays,frames)))
        else:
            yield tuple((np.asarray(frame - w, dtype = dtype) for w, frame in zip(arrays,frames)))            

def normalize_video(video, inplace = False, dtype = None):
    """Normalizes each frame in video to the mean value (intensity)
    
    Parameters
    ----------
    video : iterable
        Input multi-frame iterable object. Each element of the iterable is a tuple
        of ndarrays (frames)
    inplace : bool, optional
        Whether tranformation is performed inplace or not. 
    dtype : numpy dtype
        If specifed, determines output dtype. Only valid if inplace == False.
                
    Returns
    -------
    video : iterator
        A multi-frame iterator
    """
    for frames in video:
        if inplace == True:
            yield tuple((np.divide(frame, frame.mean(), frame) for frame in frames))
        else:
            yield tuple((np.asarray(frame / frame.mean(), dtype = dtype) for frame in frames))   

def multiply(x,y, inplace = False, dtype = None):
    """Multiplies each of the frames in multi-frame video with a given array.
    
    Parameters
    ----------
    x,y : iterable
        Input multi-frame iterable object. Each element of the iterable is a tuple
        of ndarrays (frames)
    inplace : bool, optional
        Whether tranformation is performed inplace or not. 
    dtype : numpy dtype
        If specifed, determines output dtype. Only valid if inplace == False.
        
    Returns
    -------
    video : iterator
        A multi-frame iterator
    """
    for frames, arrays in zip(x,y):
        if len(frames) != len(arrays):
            raise ValueError("Number of frames in x and y do not match")
        if inplace == True:
            yield tuple((np.multiply(frame, w, frame) for w, frame in zip(arrays,frames)))
        else:
            yield tuple((np.asarray(frame*w, dtype = dtype) for w, frame in zip(arrays,frames)))
            
class ImageShow():
    """A simple interface for video visualization using matplotlib or opencv.
    
    To use cv2 (which is much faster) for visualization 
    you must set it with :func:`.conf.set_cv2`
    """
    
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
        
        cv2.imshow(self.fig,im)
        
    def show(self, im):
        """Shows image
        
        Parameters
        ----------
        im : ndarray
            A 2D array 
        """
        if CDDMConfig.cv2 == True:
            self._cv_imshow(im)
        else:
            self._mpl_imshow(im)
            
    def __del__(self):
        if CDDMConfig.cv2 == True:
            cv2.destroyWindow(self.fig)
        else:
            plt.close()       
    
def pause(i = 1):
    """Pause in milliseconds needed to update matplotlib or opencv figures"""
    if CDDMConfig.cv2 == False:
        plt.pause(i/1000.)  
    else:
        cv2.waitKey(int(i))

#placehold for imshow figures        
_FIGURES = {}
             
def play(video, fps = 100., max_delay = 0.1):
    """Plays video for real-time visualization. 
    
    You must first call show functions (e.g. :func:`show_video`) to specify 
    what needs to be played. This function performs the actual display when in
    a for loop
    
    Parameters
    ----------
    video : iterable
        A multi-frame iterable object. 
    fps : float
        Expected FPS of the input video. If rendering of video is too slow
        for the expected frame rate, frames will be skipped to assure the 
        expected acquisition. Therefore, you must match exactly the acquisition
        frame rate with this parameter.
    max_delay : float
        Max delay that visualization can produce before it starts skipping frames.
        
    Returns
    -------
    video : iterator
        A multi-frame iterator    
        
    Examples
    --------
    
    First create some test data of a dual video
    
    >>> video = random_video(count = 256, dual = True)
    >>> video = show_video(video)
    
    Now we can load video to memory, and play it as we load frame by frame...
    
    >>> v1,v2 = asarrays(play(video, fps = 30),count = 256)
    
    """
    t0 = None
    for i, frames in enumerate(video):   
        if t0 is None:
            t0 = time.time()
            
        if time.time()-t0 < i/fps + max_delay:
            for key in list(_FIGURES.keys()):
                (viewer, queue) = _FIGURES.get(key)
                if not queue.empty():
                    viewer.show(queue.get())
                    queue.task_done()
            pause()
            
        yield frames
        
    _FIGURES.clear()
    
def figure_title(name):
    """Generate a unique figure title"""
    i = len(_FIGURES)+1
    return "Fig.{}: {}".format(i, name)

def show_video(video, id = 0, title = None):
    """Returns a video and performs image live video show.
    This works in connection with :func:`play` that does the actual display.
    
    Parameters
    ----------
    video : iterator
        A multi-frame iterator
    id : int
        Frame index
    title : str
        Unique title of the video. You can use :func:`figure_title`
        a to produce unique name.
    
    Returns
    -------
    video : iterator
        A multi-frame iterator
    """
    if title is None:
        title = figure_title("video - camera {}".format(id))

    viewer = ImageShow(title)
    queue = Queue(1)
    _FIGURES[title] = (viewer, queue)
    
    for frames in video:
        if queue.empty():
            queue.put(frames[id],block = False)
        yield frames  
     
def show_diff(video, title = None):
    """Returns a video and performs image difference live video show.
    This works in connection with :func:`play` that does the actual display.
    
    Parameters
    ----------
    video : iterator
        A multi-frame iterator
    title : str
        Unique title of the video. You can use :func:`figure_title`
        a to produce unique name.
        
    Returns
    -------
    video : iterator
        A multi-frame iterator
    """
    
    if title is None:
        title = figure_title("diff".format(id))

    viewer = ImageShow(title)
    queue = Queue(1)
    _FIGURES[title] = (viewer, queue)
    
    for frames in video:
        if queue.empty():
            m = 2* max(frames[0].max(),frames[1].max())
            im = frames[0]/m - frames[1]/m + 0.5
            queue.put(im,block = False)
        yield frames    

def random_video(shape = (512,512), count = 256, dtype = FDTYPE, max_value = 1., dual = False):
    """Random multi-frame video generator, useful for testing."""
    nframes = 2 if dual == True else 1 
    for i in range(count):
        yield tuple((np.asarray(np.random.rand(*shape)*max_value,dtype) for i in range(nframes)))

        
if __name__ == '__main__':
    #from cddm.conf import set_cv2
    #set_cv2(False)
    import cddm.conf
    cddm.conf.set_verbose(2)
    
    #example how to use show_video and play
    video = random_video(count = 256, dual = True)
    video = show_video(video)
    video = show_diff(video)
    
    v1,v2 = asarrays(play(video, fps = 30),count = 256)

#    #example how to use ImageShow
#    video = random_video(count = 256)
#    viewer = ImageShow()
#    for frames in video:
#        viewer.show(frames[0])
#        pause()
#        
    
    
