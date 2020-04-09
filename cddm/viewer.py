"""A simple matlotlib-based video and correlation data viewers"""
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from cddm.map import rfft2_kangle,  sector_indexmap
from cddm.multitau import normalize_multi, log_merge, log_average
from cddm.core import normalize
from cddm.print_tools import disable_prints, enable_prints
from cddm.decorators import doc_inherit, skip_runtime_error

class VideoViewer(object):
    """
    A matplotlib-based video viewer.
    
    Parameters
    ----------
    video : list-like, iterator
        A list of a tuple of 2D arrays or a generator of a tuple of 2D arrays. 
        If an iterator is provided, you must set 'n' as well. 
    count: int
        Length of the video. When this is set it displays only first 'count' frames of the video.
    id : int, optional
        For multi-frame data specifies camera index.
    title : str, optional
        Plot title.
    """

    def __init__(self, video, count = None, id = 0, title = ""):

        if count is None:
            try:
                count = len(video)
            except TypeError:
                raise Exception("You must specify count!")
        self.id = id
        self.index = 0
        self.video = video
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        plt.subplots_adjust(bottom=0.25)
        
        frame = next(iter(video)) #take first frame
        frame = self._prepare_image(frame)
        self.img = self.ax.imshow(frame, vmin = 0, cmap='gray') 

        self.axframe= plt.axes([0.1, 0.1, 0.7, 0.03])
        self.sframe = Slider(self.axframe, '', 0, count - 1, valinit=0, valstep=1, valfmt='%i')
 
        self.axnext = plt.axes([0.7, 0.02, 0.1, 0.05])
        self.bnext =  Button(self.axnext, '>') 
        
        self.axnext2 = plt.axes([0.6, 0.02, 0.1, 0.05])
        self.bnext2 =  Button(self.axnext2, '>>>') 
        
        self.axprev = plt.axes([0.1, 0.02, 0.1, 0.05])
        self.bprev =  Button(self.axprev, '<') 
        
        self.axprev2 = plt.axes([0.2, 0.02, 0.1, 0.05])
        self.bprev2 =  Button(self.axprev2, '<<<') 

        self.axstop = plt.axes([0.4, 0.02, 0.1, 0.05])
        self.bstop =  Button(self.axstop, 'Stop') 

        self.axplay = plt.axes([0.3, 0.02, 0.1, 0.05])
        self.bplay =  Button(self.axplay, 'Play')

        self.axfast = plt.axes([0.5, 0.02, 0.1, 0.05])
        self.bfast =  Button(self.axfast, 'FF') 
        
        self.playing = False
        self.step_fast = count/100
        self.step = 1
        
        def _play():
            while self.playing:
                plt.pause(0.001)
                next_frame = self.sframe.val + self.step
                if next_frame >= count:
                    self.playing = False
                else:
                    self.sframe.set_val(next_frame)        
        
        def stop(event):
            self.playing = False
            
        @skip_runtime_error
        def play(event):
            self.playing = True 
            self.step = 1
            _play()
            
        @skip_runtime_error
        def play_fast(event):
            self.playing = True 
            self.step = self.step_fast
            _play()
                    
        def next_frame(event):
            self.sframe.set_val(min(self.sframe.val + 1,count-1))

        def next_fast(event):
            self.sframe.set_val(min(self.sframe.val + self.step,count-1))

        def prev_frame(event):
            self.sframe.set_val(max(self.sframe.val - 1,0))

        def prev_fast(event):
            self.sframe.set_val(max(self.sframe.val - self.step,0))
            
        @skip_runtime_error  
        def update(val):
            i = int(self.sframe.val)
            try:
                frame = self.video[i] #assume list-like object
                self.index = i
            except TypeError:
                #assume generator
                frame = None
                if i > self.index:
                    for frame in self.video:
                        self.index += 1
                        if self.index >= i:
                            break
            if frame is not None:
                frame = self._prepare_image(frame)
                self.img.set_data(frame)
                self.fig.canvas.draw_idle()
    
        self.sframe.on_changed(update)
        
        self.bnext.on_clicked(next_frame)
        self.bstop.on_clicked(stop)
        self.bplay.on_clicked(play)
        self.bfast.on_clicked(play_fast)
        self.bnext2.on_clicked(next_fast)
        self.bprev2.on_clicked(prev_fast)
        self.bprev.on_clicked(prev_frame)
        
    def _prepare_image(self,im):
        if isinstance(im, tuple) or isinstance(im, list):
            return im[self.id]
        else:
            return im
        
    def show(self):
        """Shows video."""
        plt.show()
        

class DataViewer(object):
    """Shows correlation data in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.

    Parameters
    ----------
    norm : int, optional
        Normalization constant used in normalization
    scale : bool, optional
        Scale constant used in normalization.
    semilogx : bool
        Whether plot data with semilogx or not.
    shape : tuple of ints, optional
        Original frame shape. For non-rectangular you must provide this so
        to define k step.
    size : int, optional
        If specified, perform log_averaging of data with provided size parameter.
        If not given, no averaging is performed.

    """
    fig = None
    data = None
    background = None
    variance = None
    mask = None
    kmap = None
    anglemap = None
    indexmap = None
    k = 0
    angle = 0
    sector = 5
    kstep = 1
    
    def __init__(self, norm = None, scale = False, semilogx = True,  shape = None, size = None):
        self.norm  = norm
        self.scale = scale
        self.semilogx = semilogx
        self.shape = shape
        self.size = size
        
    def _init_map(self):
        self.kmap, self.anglemap = rfft2_kangle(self.kisize, self.kjsize, self.shape)
        self.indexmap = sector_indexmap(self.kmap, self.anglemap, self.angle, self.sector, self.kstep)
        
    def _init_fig(self):
        if self.data is None:
            raise ValueError("You must first set data to plot!")
        self._init_map()
        
        self.fig, (self.ax1,self.ax2) = plt.subplots(1,2,gridspec_kw = {'width_ratios':[3, 1]})
        
        self.fig.show()
        
        plt.subplots_adjust(bottom=0.25)
        
        self.graph = np.zeros(self.indexmap.shape)
        
        max_graph_value = max(self.kmap[0,:].max(), self.kmap[:,0].max()) 
        self.graph[0,0] = max_graph_value 
        
        self._max_graph_value = max_graph_value
        
        
        self.im = self.ax2.imshow(np.fft.fftshift(self.graph,0), 
            extent=[0,self.graph.shape[1],self.graph.shape[0]//2+1,-self.graph.shape[0]//2-1])
       

        self.kax = plt.axes([0.1, 0.15, 0.65, 0.03])
        self.kindex = Slider(self.kax, "k",0,int(self.kmap.max()),valinit = self.k, valfmt='%i')

        self.angleax = plt.axes([0.1, 0.10, 0.65, 0.03])
        self.angleindex = Slider(self.angleax, "angle",-90,90,valinit = self.angle, valfmt='%.2f')        

        self.sectorax = plt.axes([0.1, 0.05, 0.65, 0.03])
        self.sectorindex = Slider(self.sectorax, "sector",0,180,valinit = self.sector, valfmt='%.2f') 
                                  
        def update(val):
            self.set_mask(int(round(self.kindex.val)),self.angleindex.val,self.sectorindex.val, self.kstep)
            self.plot()
            
        self.kindex.on_changed(update)
        self.angleindex.on_changed(update)
        self.sectorindex.on_changed(update)
        
        
    def set_data(self, data, background = None, variance = None):
        """Sets correlation data.
        
        Parameters
        ----------
        data : tuple
            A data tuple (as computed by ccorr, cdiff, adiff, acorr functions)
        background : tuple or ndarray
            Background data for normalization. For adiff, acorr functions this
            is ndarray, for cdiff,ccorr, it is a tuple of ndarrays.
        variance : tuple or ndarray
            Variance data for normalization. For adiff, acorr functions this
            is ndarray, for cdiff,ccorr, it is a tuple of ndarrays.
        """
        self.data = data
        self.background = background
        self.variance = variance
        self.kisize, self.kjsize = data[0].shape[:-1]
        
    def get_data(self):
        """Returns computed k-averaged data and time
        
        Returns
        -------
        x, y : ndarray, ndarray
            Time, data tuple of ndarrays.
        """
        if self.data is None:
            raise ValueError("No data, you must first set data.")
        if self.mask is None:
            raise ValueError("Mask not specified, you must first set mask.")
            
        data = normalize(self.data, self.background, self.variance, norm = self.norm, scale = self.scale, mask = self.mask)
        if self.size is not None:
            t, data = log_average(data, self.size)
            avg_data = data.mean(0) 
        else:
            avg_data = data.mean(0)
            t = np.arange(len(avg_data))
        return t, avg_data
    
    def get_k(self):
        """Returns average k value of current data."""
        if self.mask is None:
            raise ValueError("Mask not specified, you must first set mask.")
        return self.kmap[self.mask].mean()
    
    def set_mask(self, k, angle = 0, sector = 5, kstep = 1):
        """Sets k-mask for averaging,
        
        Parameters
        ----------
        k : int
            k index in kstep units.
        angle : int
            Mean k-angle in degrees. Measure with respecto to image horizontal axis.
        sector : int
            Averaging full angle in degrees.
        kstep : float, optional
            K step in units of minimum k step for a given FFT dimensions.
            
        Returns
        -------
        ok : bool
            True if mask is valid else False
        """
        self.k = k
        self.angle = angle
        self.kstep = kstep
        self.sector = sector
        if self.kmap is None:
            self._init_map()
        self.indexmap = sector_indexmap(self.kmap, self.anglemap, self.angle, self.sector, self.kstep)
        self.mask = (self.indexmap == int(self.k))
        return self.mask.any()
        
    def plot(self):
        """Plots data. You must first call :meth:`.set_data` to set input data"""
        
        if self.fig is None:
            self._init_fig()
            self.set_mask(int(round(self.k)),self.angle,self.sector, self.kstep)
            
        nans = (self.indexmap == -1)
        self.graph = self.indexmap*1.0 # make it float
        self.graph[self.mask] = self._max_graph_value +1
        self.graph[nans] = np.nan
        
        self.im.set_data(np.fft.fftshift(self.graph,0))
                    
        with np.errstate(divide='ignore', invalid='ignore'):
            state = disable_prints()
            t, avg_data = self.get_data()
            enable_prints(state)            
        try:
            self.l.set_ydata(avg_data)
            self.l.set_xdata(t)
            self.ax1.relim()
            self.ax1.autoscale_view()
        except AttributeError:
            if self.semilogx == True:
                self.l, = self.ax1.semilogx(t,avg_data)
            else:
                self.l, = self.ax1.plot(t,avg_data)

        self.fig.canvas.draw() 
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def show(self):
        """Shows plot."""
        plt.show()
        #self.fig.show()
    
class MultitauViewer(DataViewer):
    """Shows multitau data in plot. You need to hold reference to this object, 
    otherwise it will not work in interactive mode.
    
    Parameters
    ----------
    norm : int, optional
        Normalization constant used in normalization
    scale : bool, optional
        Scale constant used in normalization.
    semilogx : bool
        Whether plot data with semilogx or not.
    shape : tuple of ints, optional
        Original frame shape. For non-rectangular you must provide this so
        to define k step.
    """
    def __init__(self, norm = None, scale = False, semilogx = True,  shape = None):

        self.norm  = norm
        self.scale = scale
        self.semilogx = semilogx
        self.shape = shape
        
    @doc_inherit
    def set_data(self, data, background = None, variance = None):
        self.data = data
        self.background = background
        self.variance = variance
        self.kisize, self.kjsize = data[0][0].shape[:-1]
    
    @doc_inherit    
    def get_data(self):
        if self.data is None:
            raise ValueError("No data, you must first set data.")
        if self.mask is None:
            raise ValueError("Mask not specified, you must first set mask.")
            
        data = normalize_multi(self.data, self.background, self.variance, norm = self.norm, scale = self.scale, mask = self.mask)
        t, data = log_merge(*data)
        avg_data = data.mean(0) 
        return t, avg_data
        

       
if __name__ == "__main__":
    video = (np.random.randn(256,256) for i in range(256))
    vg = VideoViewer(video, 256, title = "iterator example") #must set nframes, because video has no __len__
    vg.show()
    
    video = [np.random.randn(256,256) for i in range(256)] 
    vl = VideoViewer(video, title = "list example") 
    vl.show()  

    