"""
Video processing tools. 

You can use this helper functions to perform normalization, background subtraction 
and windowing on multi-frame data. 

There are also function for real-time display of videos for real-time analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import time, os
from cddm.conf import CDDMConfig, FDTYPE
from cddm.print_tools import print_progress, print1, print_frame_rate

from cddm.fft import _rfft2
from cddm.buffer import buffered
from cddm.run import asrunning
from cddm.viewer import FramesViewer, figure_title
from cddm.decorators import deprecated

def user_warning(message):
    import warnings
    warnings.warn(message, UserWarning, stacklevel=2)

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
    elif not count > 0:
        raise ValueError("Count must be greater than 0")

    print_progress(0, count)
    
    video = asrunning(video)
    
    video = iter(video)
    
    # read first element to determine shape and type
    frames = next(video)
    out = tuple((np.empty(shape = (count,) + frame.shape, dtype = frame.dtype) for frame in frames))
    
    # now load data
    [_load(out[i][0],frame) for i,frame in enumerate(frames)]
    #in case video is empty, we have n defined
    n = 0
    for j,frames in enumerate(video):
        n = j+1
        if n == count:
            # in case we load less than full video, just stop.
            break
        print_progress(n, count)
        [_load(out[i][n],frame) for i,frame in enumerate(frames)]
        
    if n < count - 1:
        raise ValueError("Input video too short for a given count")
        
    print_progress(count, count)
    print_frame_rate(count,t0)
    return out

def asmemmaps(basename, video, count = None):
    """Loads multi-frame video into numpy memmaps. 
    
    Actual data is written to numpy files with the provided basename and
    subscripted by source identifier (index), e.g. "{basename}_0.npy" and "{basename}_1.npy"
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

def open_memmaps(basename):
    """Opens video stored as numpy a arrays into memmaps.
    
    Returns
    -------
    out : tuple of arrays
        A tuple of memmapped array(s) representing video(s)
    """
    files = (basename+"_0.npy", basename+"_0.npy")
    arrays = tuple((np.lib.format.open_memmap(fname) for fname in files if os.path.exists(fname)))
    return arrays

def frommemmaps(basename):
    """Opens video stored as numpy arrays and returns a video iterator.
    
    Returns
    -------
    out : tuple
        A video iterable. A tuple of multi-frame data (arrays) 
    """
    arrays = open_memmaps(basename)
    for frames in zip(*arrays):
        yield frames

def recorded(basename,video, count = None):
    """Creates a recording video. Video is saved to disk as numpy files during
    iteration over frames.
    
    Returns
    -------
    out : tuple
        A video iterable. A tuple of multi-frame data (arrays) 
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
    
    frames = next(video)
    out = _empty_arrays(frames)
    [_load(out[i][0],frame) for i,frame in enumerate(frames)]
    yield frames
    
    for j,frames in enumerate(video):
        [_load(out[i][j+1],frame) for i,frame in enumerate(frames)]
        yield frames
    
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
           
class SourceIterable():
    """Source data holder for iterables. This object is used in conjunction 
    with :class:`StreamingIterable`. See also :func:`split`.
    """
    
    def __init__(self, iterable, n = 2):
        self.source = iter(iterable)
        self.stream_index = [0] * n
        self.stream_data = [{} for i in range(n)] #each dict must be unique cannot use [{}]*n!
        self.source_index = 0
        self.count = None
        
    def next_data(self,stream):
        index = self.stream_index[stream]
        if index == self.source_index:
            try:
                out = next(self.source)
                for data in self.stream_data:
                    data[self.source_index] = out
                self.source_index += 1
            except StopIteration:
                self.count = index
        
        if index == self.count:
            raise StopIteration
        else:
            out = self.stream_data[stream].pop(index)     
            self.stream_index[stream] = index + 1
            return out

class StreamingIterable():
    """An iterable video that works with :class:`SourceIterable` input data types."""
    def __init__(self, iterable, stream = 0):
        if not hasattr(iterable,"next_data"):
            raise ValueError("Invalid iterable data. You must provide a :class:`SourceIterable`-like object.")
        self.iterable = iterable
        self.stream = stream
        
    def __next__(self):
        return self.iterable.next_data(self.stream)
           
    def __iter__(self):
        return self

def split(iterable, n = 2):
    """Splits an iterable into two or more iterables.
    
    Parameters
    ----------
    iterable : iterator
        Input iterator, or any iterable object.
    n : int
        Number of streams that you wish to split input data to.
    
    Returns
    -------
    out : tuple of iterables
        A tuple of iterable objects.
        
    Examples
    --------
    
    Works on any iterable object, so create one
    
    >>> v = range(10)
    
    Now we can split and create two identical iterators
    >>> v1, v2 = split(v, n = 2)
    >>> list(v1)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> list(v2)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    Now that we have read the data, the iterables are empty
    
    >>> list(v1)
    []
    >>> list(v2)
    []
    
    """
    source = SourceIterable(iterable, n)
    return tuple((StreamingIterable(source,i) for i in range(n)))

def apply(video, func):
    """Apply a custom function to frames

    Parameters
    ----------
    video : iterable
        A multi-frame iterable object. 
    func : callable
        A callable that takes two arguments, an integer describing the current index
        and a multi-frame data. The function must return a processed -multi-frame data
    Returns
    -------
    video : iterator
        A multi-frame iterator   
    """    
    for i, frames in enumerate(video):
        yield func(i,frames)


@deprecated("Use `running` instead.")   
def play(video, fps = None, max_delay = None):
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
    from cddm.run import run_buffered
    from cddm.viewer import pause
    
    return run_buffered(video, fps = fps, spawn = False, finalize = pause)

@deprecated("Use `running` instead.")  
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
    from cddm.run import run_buffered
    from cddm.viewer import pause
    
    return run_buffered(video, fps = fps, spawn = True, finalize = pause)

def show_frames(video, title = None, viewer = None, selected = None, **kwargs):
    """
    Creates a showing video. Note that the returned video is unchanged. 
    With the optional parameters you can tune the visualization parameters.
    Note that this function only creates a visualizer for display.
    
    To perform the actual display, you have to create a running video instance.
    
    Parameters
    ----------
    video : iterator
        A multi-frame iterator
    title : str
        Unique title of the video. You can use :func:`figure_title`
        a to produce unique name.
    viewer: callable
        Here you can provide your own callable viewer object that is 
        responsible for the visualization and data conversion.
    selected : callable
        A callable that can be set to define which frames to process.
        
    The rest of the parameters are passed directly to the FramesViewer.
        
    in_space : str
        One of "image", "rfft2", "fft2", describing input image space.
    out_space : str
        One of "image", "rfft2", "fft2", describing output image space.
    repr_mode : str
        One of "real", "imag", "abs2" or "phase" describing the representation
        mode of the (complex) data.  
    typ : str
        Frames converter type, one of "cam1", "cam2", "diff" or "corr"
    normalize : bool
        Whether to normalize each frame with mean intensity or not.
    auto_background : bool
        Whether to perform automatic background subtraction.
    navg : int
        How many frames to process in the averaging of the background. If set
        to zero, it take all acquired frames.
    """
  
    if title is None:
        typ = kwargs.get("typ", "cam1")
        title = figure_title(f"frames - {typ}")
    if viewer is None:
        viewer = FramesViewer(title, **kwargs)
    return buffered(video, maxsize=1, callback = viewer, selected = selected)
        
            
def show_data(iterable, viewer, selected = None):
    """Show iterable data. You must provide a valid viewer. 
    
    This function is not only for video, but for any kind of data. You must
    define a viewer, which implements a self.show(data) method, which is responsible
    for the actual data show.
    
    Parameters
    ----------
    iterable : iterator
        Any kinf of data iterable.
    viewer : any
        Any viewer-like object with a show() method.
        
    Returns
    -------
    iterable : iterator
        A data iterable.
    """

    if not callable(viewer):
        raise ValueError("Invalid viewer.")
        
    return buffered(iterable, maxsize=1, callback = viewer, selected = selected)

def random_video(shape = (512,512), count = 256, dtype = FDTYPE, max_value = 1., dual = False):
    """Random multi-frame video generator, useful for testing."""
    nframes = 2 if dual == True else 1 
    for i in range(count):
        time.sleep(0.01)
        yield tuple((np.asarray(np.random.rand(*shape)*max_value,dtype) for i in range(nframes)))


if __name__ == '__main__':
    
    from cddm.run import running   
    import cddm.conf
    cddm.conf.set_verbose(2)


    cddm.conf.set_showlib("pyqtgraph")    
    #example how to use show_video and play
    video = random_video(count = 16*16, dual = True)
    video = recorded("deleteme",video, count = 16*16)
    #video = load(video, 1256)
    video = show_frames(video, typ = "cam1", )
    

    video = show_frames(video, typ = "diff")
    
    #p = play_threaded(video)
    #v1,v2 = asarrays(video,count = 16*16)
    #v1,v2 = asarrays(play(video),count = 16*16)
    with running(video) as video:
        v1,v2 = asarrays(video,count = 16*16)
        
##    #example how to use ImageShow
##    video = random_video(count = 256)
##    viewer = ImageShow()
##    for frames in video:
##        viewer.show(frames[0])
##        pause()
##        
#    
    
