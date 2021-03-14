"""
Video processing tools. 

You can use this helper functions to perform normalization, background subtraction 
and windowing on multi-frame data. 

There are also function for real-time display of videos for real-time analysis.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import matplotlib.pyplot as plt
import time   
from cddm.conf import CDDMConfig, CV2_INSTALLED,  FDTYPE, PYQTGRAPH_INSTALLED
from cddm.print_tools import print_progress, print1, print_frame_rate

from cddm.fft import _rfft2

try:
    from queue import Queue
except ImportError:
    #python 2.7
    from Queue import Queue

import threading

if CV2_INSTALLED:
    import cv2
    
if PYQTGRAPH_INSTALLED:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui
    pg.setConfigOptions(imageAxisOrder = "row-major")

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
        try:
            count = len(video)
        except TypeError:
            raise ValueError("You must provide count")
        
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

def load(video, count = None):
    """Loads video into memory. 
     
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
    out : tuple
        A video iterable. A tuple of multi-frame data (arrays) 
    """
    video = asarrays(video, count)
    return tuple(fromarrays(video))

def crop(video, roi = (slice(None), slice(None))):
    """Crops each frame in the video. 
    
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

def mask(video, mask = None):
    """Masks each frame in the video. 
    
    Parameters
    ----------
    video : iterable
        Input multi-frame iterable object. Each element of the iterable is a tuple
        of ndarrays (frames)
    mask : ndarrray
        A boolean index array for masking (boolean indexing).
        
    Returns
    -------
    video : iterator
        A multi-frame iterator
    """
    for frames in video:
        yield tuple((frame[mask] for frame in frames))

def subtract(x, y, inplace = False, dtype = None):
    """Subtracts two videos.
    
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

def add(x, y, inplace = False, dtype = None):
    """Adds two videos.
    
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
            yield tuple((np.add(frame, w, frame) for w, frame in zip(arrays,frames)))
        else:
            yield tuple((np.asarray(frame + w, dtype = dtype) for w, frame in zip(arrays,frames)))            


def normalize_video(video, inplace = False, dtype = None):
    """Normalizes each frame in the video to the mean value (intensity).
    
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
    """Multiplies two videos.
    
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
    """A simple interface for video visualization using matplotlib, opencv, or
    pyqtgraph.
    
    Parameters
    ----------
    title : str
       Title of the video
    norm_func : callable
        Normalization function that takes a single argument (array) and returns
        a single element (array). Can be used to apply custom normalization 
        function to the image before it is shown.
    """
    
    fig = None
    
    def __init__(self, title = "video", norm_func = lambda x : x):
        self.title = title
        self._prepare_image = norm_func
    
    def _pg_imshow(self,im):
        im = self._prepare_image(im)
        if self.fig is None:
            self.im = pg.image(im, title = self.title)
            self.fig = self.im.window()
        else:
            self.im.setImage(im)
 
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
        if im.max() > 1:
            im = im / im.max()
        
        cv2.imshow(self.fig,im)
        
    def show(self, im):
        """Shows image
        
        Parameters
        ----------
        im : ndarray
            A 2D array 
        """
        if CDDMConfig.showlib == "cv2":
            self._cv_imshow(im)
        elif CDDMConfig.showlib == "pyqtgraph":
            self._pg_imshow(im)
        else:
            self._mpl_imshow(im)
            
    def __del__(self):
        if self.fig is not None:
            if CDDMConfig.showlib == "cv2":
                cv2.destroyWindow(self.fig)
            elif CDDMConfig.showlib == "pyqtgraph":
                self.fig.destroy()
            else:
                plt.close(self.fig)       
    
def pause(i = 1):
    """Pause in milliseconds needed to update matplotlib or opencv figures
    For pyqtgraph, it performs app.processEvents()"""
    if CDDMConfig.showlib == "cv2":
        cv2.waitKey(int(i))
    elif CDDMConfig.showlib == "matplotlib":
        plt.pause(i/1000.)
    else: 
        app = QtGui.QApplication.instance()
        if app is None:
            app = QtGui.QApplication([])
        app.processEvents()
        
#placehold for imshow figures        
_FIGURES = {}
             
def play(video, fps = 100., max_delay = 0.1):
    """Plays video for real-time visualization. 
    
    You must first call show functions (e.g. :func:`show_video`) to specify 
    what needs to be played. This function performs the actual display when in
    a for loop.
    
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
    
    See Also
    --------
    :func:`.video.play_threaded`
    
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
    

def play_threaded(video, fps = None):
    """Plays video for real-time visualization. 
    
    You must first call show functions (e.g. :func:`show_video`) to specify 
    what needs to be played. This function performs the actual display when in
    a for loop. It works similar to :func:`play`, but it first creates a thread
    and starts the video iterator in the background. 
    
    Parameters
    ----------
    video : iterable
        A multi-frame iterable object. 
    fps : float, optional
        Desired video fps for display. This may be different from the actual 
        video fps. If not set, it will display video as fast as possible.
        
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
    
    >>> v1,v2 = asarrays(play_threaded(video, fps = 30),count = 256)

    See Also
    --------
    :func:`.video.play`
    
    """
    q = Queue()
    
    def worker(video):
        try:
            for frames in video: 
                q.put(frames)
        finally:
            q.put(None)
    
    threading.Thread(target=worker,args = (video,) ,daemon=True).start()
    out = False #dummy  
    t0 = None   
    i = 0
    while True:
        if fps is None:
            while not q.empty():
                out = q.get()
                q.task_done()
                if out is None:
                    break
                else:
                    yield out
        else:
            while True:
                out = q.get()
                q.task_done()
                if t0 is None:
                    t0 = time.time()
                if out is None:
                    break
                else:
                    yield out
                if time.time()-t0 >= i/fps:
                    break            
                
        if out is None:
            break

        for key in list(_FIGURES.keys()):
            (viewer, queue) = _FIGURES.get(key)
            if not queue.empty():
                viewer.show(queue.get())
                queue.task_done()
        i+=1
        pause()
            
    _FIGURES.clear()
        
def figure_title(name):
    """Generate a unique figure title"""
    i = len(_FIGURES)+1
    return "Fig.{}: {}".format(i, name)

def norm_rfft2(clip = None, mode = "real"):
    """Returns a frame normalizing function for :func:`show_video`"""
    if mode not in ("real", "imag", "abs"):
        raise ValueError("Wrong mode")
        
    def _clip_fft(im):
        im = _rfft2(im)
        if mode == "real":
            im = im.real
        elif mode == "imag":
            im = im.imag
        else:
            im = np.abs(im)
        
        if clip is None:
            im[0,0] = 0
            clip_factor = im.max()
        else:
            im[0,0] = 0
            clip_factor = clip
        
        if mode == "abs":
            im = im/(clip_factor)
        else:
            im = im/(clip_factor*2) + 0.5
        im = im.clip(0,1)  
        return np.fft.fftshift(im,0)  
    return _clip_fft

def show_fft(video, id = 0, title = None, clip = None, mode = "abs"):
    """Show fft of the video.
    
    Parameters
    ----------
    video : iterator
        A multi-frame iterator
    id : int
        Frame index
    title : str, optional
        Unique title of the video. You can use :func:`.video.figure_title`
        to create a unique name.
    clip : float, optional
        Clipping value. If not given, it is determined automatically.
    mode : str, optional
        What to display, "real", "imag" or "abs"
    
    Returns
    -------
    video : iterator
        A multi-frame iterator
    
    """
    if title is None:
        title = figure_title("fft - camera {}".format(id))
    norm_func = norm_rfft2(clip, mode)
    return show_video(video, id = id, title = title, norm_func = norm_func)
               

def show_video(video, id = 0, title = None, norm_func = lambda x : x.real):
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
    norm_func : callable
        Normalization function that takes a single argument (array) and returns
        a single element (array). Can be used to apply custom normalization 
        function to the image before it is shown.
    
    Returns
    -------
    video : iterator
        A multi-frame iterator
    """
    if title is None:
        title = figure_title("video - camera {}".format(id))

    viewer = ImageShow(title, norm_func)
    queue = Queue(1)
    _FIGURES[title] = (viewer, queue)
    
    for frames in video:
        if queue.empty():
            queue.put(frames[id],block = False)
        yield frames  
     
def show_diff(video, title = None, normalize = False, dt = None, t1 = None, t2 = None):
    """Returns a video and performs image difference live video show.
    This works in connection with :func:`play` that does the actual display.
    
    Parameters
    ----------
    video : iterator
        A multi-frame iterator
    title : str
        Unique title of the video. You can use :func:`figure_title`
        a to produce unique name.
    normalize : bool
        Whether to normalize frames to its mean value before subtracting. Note
        that this does not normalize the output video, only the displayed video
        is normalized.
        
    Returns
    -------
    video : iterator
        A multi-frame iterator
    """
    def _process_frames(frames, queue):
        if queue.empty():
            x,y = frames
            x,y = x.real, y.real
            if normalize == True:
                x = x/x.mean()
                y = y/y.mean()
            m = 2* max(x.max(),y.max())
            im = x/m - y/m + 0.5
            queue.put(im,block = False)        
    
    if title is None:
        title = figure_title("difference video")

    viewer = ImageShow(title)
    queue = Queue(1)
    _FIGURES[title] = (viewer, queue)
    
    if dt is None:
        for frames in video:
            _process_frames(frames, queue)
            yield frames 
    else:
        for frames,_t1,_t2 in zip(video,t1,t2):
            if abs(_t2-_t1) in dt:
                _process_frames(frames, queue)
            yield frames         
        

def random_video(shape = (512,512), count = 256, dtype = FDTYPE, max_value = 1., dual = False):
    """Random multi-frame video generator, useful for testing."""
    nframes = 2 if dual == True else 1 
    for i in range(count):
        time.sleep(0.02)
        yield tuple((np.asarray(np.random.rand(*shape)*max_value,dtype) for i in range(nframes)))

        
if __name__ == '__main__':
    
    
    import cddm.conf
    cddm.conf.set_verbose(2)


    cddm.conf.set_showlib("pyqtgraph")    
    #example how to use show_video and play
    video = random_video(count = 1256, dual = True)
    #video = load(video, 1256)
    video = show_video(video)
    video = show_diff(video)

    #p = play_threaded(video)
    #v1,v2 = asarrays(video,count = 1256)
    #v1,v2 = asarrays(play(video,fps = 50),count = 1256)
    v1,v2 = asarrays(play_threaded(video),count = 1256)
##    #example how to use ImageShow
##    video = random_video(count = 256)
##    viewer = ImageShow()
##    for frames in video:
##        viewer.show(frames[0])
##        pause()
##        
#    
    
