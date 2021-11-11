"""A simple matlotlib-based video and correlation data viewers"""
from __future__ import absolute_import, print_function, division
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, AxesWidget


from cddm.map import fft2_kangle, fft2_sector_indexmap, as_fft2_shape, from_rfft2_data, as_rfft2_mask, as_rfft2_conj
from cddm.multitau import normalize_multi, log_merge, log_average
from cddm.norm import normalize, _default_norm_from_data, _method_from_data, \
    NORM_STRUCTURED, NORM_SUBTRACTED, NORM_WEIGHTED, NORM_STANDARD, NORM_COMPENSATED, \
        fold_data, CALC_MEAN, CALC_SQUARE, AVAILABLE_CORR_NORMS,AVAILABLE_DIFF_NORMS

from cddm.print_tools import disable_prints, enable_prints
from cddm.decorators import skip_runtime_error
from cddm.conf import CDDMConfig, CV2_INSTALLED, PYQTGRAPH_INSTALLED
from cddm.frames import FramesConverter

_matplotlib_3_4_or_greater = False

try:
    import matplotlib
    major, minor = matplotlib.__version__.split(".")[0:2]
    if int(major) >= 3 and int(minor) >=4:
        _matplotlib_3_4_or_greater = True
except:
    print("Could not determine matplotlib version you are using, assuming < 3.4")


if CV2_INSTALLED:
    import cv2
    
if PYQTGRAPH_INSTALLED:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui
    pg.setConfigOptions(imageAxisOrder = "row-major")

_FIGURES = set()

def figure_title(name):
    """Generate a unique figure title"""
    i = len(_FIGURES)+1
    title = "Fig.{}: {}".format(i, name)
    return title

        
class CustomRadioButtons(RadioButtons):

    def __init__(self, ax, labels, active=0, activecolor='C0', size=49,
                 orientation="horizontal", **kwargs):
        """
        Add radio buttons to an `~.axes.Axes`.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : color
            The color of the selected button.
        size : float
            Size of the radio buttons
        orientation : str
            The orientation of the buttons: 'vertical' (default), or 'horizontal'.
        Further parameters are passed on to `Legend`.
        """
        AxesWidget.__init__(self, ax)
        self.activecolor = activecolor
        axcolor = ax.get_facecolor()
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        circles = []
        for i, label in enumerate(labels):
            if i == active:
                self.value_selected = label
                facecolor = activecolor
            else:
                facecolor = axcolor
            p = ax.scatter([],[], s=size, marker="o", edgecolor='black',
                           facecolor=facecolor)
            circles.append(p)
        if orientation == "horizontal":
            kwargs.update(ncol=len(labels), mode="expand")
        kwargs.setdefault("frameon", False)    
        self.box = ax.legend(circles, labels, loc="center", **kwargs)
        self.labels = self.box.texts
        self.circles = self.box.legendHandles
        for c in self.circles:        
            c.set_picker(5)
            
        if _matplotlib_3_4_or_greater:
            self._observers = matplotlib.widgets.cbook.CallbackRegistry()
            
        else:
            self.cnt = 0
            self.observers = {}

        self.connect_event('pick_event', self._clicked)


    def _clicked(self, event):
        if (self.ignore(event) or event.mouseevent.button != 1 or
            event.mouseevent.inaxes != self.ax):
            return
        if event.artist in self.circles:
            self.set_active(self.circles.index(event.artist))

#: available complex correlation data representation modes
AVAILABLE_DATA_REPR_MODES = ("real", "imag", "abs", "phase")

def _inspect_data_repr_mode(mode):
    mode = str(mode).strip().lower()
    if mode not in AVAILABLE_DATA_REPR_MODES:
        raise ValueError("Invalid data representation mode. Must be one of {}".format(AVAILABLE_DATA_REPR_MODES))
    return mode  
            
def data_repr(data, mode = "real"):
    mode = _inspect_data_repr_mode(mode)
    if mode == "real":
        return data.real
    elif mode == "imag":
        return data.imag
    elif mode == "abs":
        return np.abs(data)
    elif mode == "phase":
        return np.arctan2(data.imag, data.real)
    
class DataArrayViewer(object):
    fig = None
    data = None
    mode = "real"
    
    def __init__(self, semilogx = True, axes = (),):
        self.semilogx = semilogx
        self.axes = tuple(axes)
        self._default_indices = (0,) * len(axes)
        self.set_indices(*self._default_indices)
   
    def _init_fig(self):
        if self.data is None:
            raise ValueError("You must first set data to plot!")

        self.fig, self.ax = plt.subplots(1,1)
        self.fig.show()
        plt.subplots_adjust(bottom=0.25)

        self.selectorax = plt.axes([0.1, 0.95, 0.8, 0.03])
        self.selectorindex = CustomRadioButtons(self.selectorax, ["real", "imag", "abs", "phase"], active = 0)
        
        self.slider_axes = {}
        self.sliders = {}
        
        def update_slider(val):
            self.set_indices(*tuple((int(self.sliders[ax].val) for ax in self.axes)))
            self.plot()      
        
        for i,ax in enumerate(self.axes):
            self.slider_axes[ax] = plt.axes([0.1, 0.05 + i*0.15/len(self.axes), 0.8, 0.03])
            size = self.shape[ax]-1
            self.sliders[ax] = Slider(self.slider_axes[ax], "axis {}".format(ax) ,-size, size,valinit = self._default_indices[i], valfmt='%i')
            self.sliders[ax].on_changed(update_slider)

        def update_mode(val):
            self.mode = val
            update_slider(val)

        self.selectorindex.on_clicked(update_mode)

    def set_indices(self,*vals):
        indices = [slice(None)] * len(self.axes)
        for i,ax in enumerate(self.axes):
            indices[ax] = vals[i]
        self.indices = tuple(indices)
              
    def set_data(self, data, t = None):
        """Sets correlation data.
        
        Parameters
        ----------
        data : ndarray
            Normalized data.
        """
        #first try to see if data compatible with the defined axes
        axes = range(data.ndim)
        try:
            for ax in self.axes:
                axes[ax]
        except IndexError:
            raise ValueError("Input data not compatible with defined axes.")
        
        self.data = data
        self.shape = self.data.shape

        t =  np.asarray(t) if t is not None else np.arange(data.shape[-1])
        if len(t) != data.shape[-1]:
            raise ValueError("Wrong time array length")
        else:
            self.t = t
            
    def _get_avg_data(self):
        data = self.data[self.indices]
        avg = data.mean(axis = tuple(range(data.ndim-1)))
        return self.t, data_repr(avg, self.mode)
          
    def get_data(self):
        """Returns computed k-averaged data and time
        
        Returns
        -------
        x, y : ndarray, ndarray
            Time, data ndarrays.
        """
        
        if self.data is None:
            raise ValueError("No data, you must first set data.")

        return self._get_avg_data()
            

    def plot(self):
        """Plots data. You must first call :meth:`.set_data` to set input data"""
        
        if self.fig is None:
            self._init_fig()
              
        with np.errstate(divide='ignore', invalid='ignore'):
            state = disable_prints()
            t, avg_data = self.get_data()
            enable_prints(state)            
        try:
            self.l.set_ydata(avg_data)
            self.l.set_xdata(t)
            self.ax.relim()
            self.ax.autoscale_view()
        except AttributeError:
            if self.semilogx == True:
                self.l, = self.ax.semilogx(t,avg_data)
            else:
                self.l, = self.ax.plot(t,avg_data)

        self.fig.canvas.draw() 
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def show(self,data = None):
        """Sets data and plots."""
        if data is not None:
            self.set_data(data)
        self.plot()
        
    def close(self):
        pass
    
    def __call__(self, i, data):
        return self.show(data = data)
        
class CorrArrayViewer(DataArrayViewer):
    """Plots raw correlation data. You need to hold reference to this object, 
    otherwise it will not work in interactive mode.

    Parameters
    ----------
    semilogx : bool
        Whether plot data with semilogx or not.
    shape : tuple of ints, optional
        Original frame shape. For non-rectangular you must provide this so
        to define k step.
    size : int, optional
        If specified, perform log_averaging of data with provided size parameter.
        If not given, no averaging is performed.
    norm : int, optional
        Normalization constant used in normalization
    scale : bool, optional
        Scale constant used in normalization.
    mask : ndarray, optional
        A boolean array indicating which data elements were computed.
    fold : bool
        For complex data this is a required parameter. I specifies whether to 
        fold data or not.
    """
    background = None
    variance = None
    
    def __init__(self, semilogx = True,  axes = (0,), size = None, norm = None, scale = False, fold = None):
        self.axes = axes
        self._default_indices = (0,) * len(axes)
        self.set_indices(*self._default_indices)
        self.norm  = norm
        self.scale = scale
        self.semilogx = semilogx
        self.size = size
        self.fold = fold
        
    @property   
    def mask(self):
        mask = np.zeros(self.shape[:-1], bool)
        mask[self.indices] = True
        return mask
        
    def set_norm(self, value):
        """Sets norm parameter"""
        method = _method_from_data(self.data)
        self.norm = _default_norm_from_data(self.data,method,value)     
        
    def _init_fig(self):
        super()._init_fig()
        
        self.set_norm(self.norm)
        
        #self.rax = plt.axes([0.48, 0.55, 0.15, 0.3])
        self.cax = plt.axes([0.44, 0.72, 0.2, 0.15])
        
        self.active = [bool(self.norm & NORM_STRUCTURED),bool(self.norm & NORM_SUBTRACTED), bool((self.norm & NORM_WEIGHTED == NORM_WEIGHTED)) , bool((self.norm & NORM_COMPENSATED) == NORM_COMPENSATED)  ]
        
        self.check = CheckButtons(self.cax, ("structured", "subtracted", "weighted", "compensated"), self.active)
        
        #self.radio = RadioButtons(self.rax,("norm 0","norm 1","norm 2","norm 3","norm 4", "norm 5", "norm 6", "norm 7"), active = self.norm, activecolor = "gray")
 
        def update(label):
            index = ["structured", "subtracted", "weighted", "compensated"].index(label)
            status = self.check.get_status()

            norm = NORM_STRUCTURED if status[0] == True else NORM_STANDARD    
            
            if status[1]:
                norm = norm | NORM_SUBTRACTED
            if status[2]:
                norm = norm | NORM_WEIGHTED
            if status[3]:
                norm = norm | NORM_COMPENSATED
            try:
                self.set_norm(norm)
            except ValueError:
                self.check.set_active(index)
            self.plot()

                
        self.check.on_clicked(update)

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
        self.shape = self.data[0].shape
        
    def _get_avg_data(self):
        
        data = normalize(self.data, self.background, self.variance, norm = self.norm, scale = self.scale, mask = self.mask)

        if np.iscomplexobj(data) :
            if isinstance(self.background, tuple) or isinstance(self.variance, tuple) or self.fold is not None:
                data = fold_data(data)
            elif self.background is None and self.variance is None:
                raise ValueError("Cannot determine whether data folding is needed. You must explicitly define whether to fold correlation data or not.")

        if self.size is not None:
            # folding has already been made, so set to False
            t, data = log_average(data, self.size, fold = False)
        else:
            t = np.arange(data.shape[-1])

        return t, data_repr(data.mean(axis = -2), self.mode)

    def show(self,data = None):
        """Sets data and plots."""
        if data is not None:
            if len(data) == 3:
                data, bg, var = data
                self.set_data(data, bg, var )
            else:
                self.set_data(data)
        self.plot()
        

class MultitauArrayViewer(CorrArrayViewer):
    """Shows multitau data in plot. You need to hold reference to this object, 
    otherwise it will not work in interactive mode.
    
    Parameters
    ----------
    semilogx : bool
        Whether plot data with semilogx or not.
    shape : tuple of ints, optional
        Original frame shape. For non-rectangular you must provide this so
        to define k step.
    norm : int, optional
        Normalization constant used in normalization
    scale : bool, optional
        Scale constant used in normalization.
    mask : ndarray, optional
        A boolean array indicating which data elements were computed.
    fold : bool
        For complex data this is a required parameter. I specifies whether to 
        fold data or not.
    """
        
    def set_norm(self, value):
        """Sets norm parameter"""
        method = _method_from_data(self.data[0])
        self.norm = _default_norm_from_data(self.data[0],method,value)  
        
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
        self.shape = data[0][0].shape
        
        
    def _get_avg_data(self):
        data = normalize_multi(self.data, self.background, self.variance, norm = self.norm, scale = self.scale, mask = self.mask)
        
        if np.iscomplexobj(data[0]) :
            if isinstance(self.background, tuple) or isinstance(self.variance, tuple) or self.fold is not None:
                data = fold_data(data[0],1), fold_data(data[1],1)
            elif self.background is None and self.variance is None:
                raise ValueError("Cannot determine whether data folding is needed. You must explicitly define whether to fold correlation data or not.")

        if self.semilogx == True:
            t, data = log_merge(*data, fold = False)
        else:
            fast, slow = data
            t = np.arange(fast.shape[-1])
            data = fast
                    
        return t, data_repr(data.mean(axis = -2), self.mode)
    
class DataViewer(object):
    """Plots normalized correlation data. You need to hold reference to this object, 
    otherwise it will not work in interactive mode.

    Parameters
    ----------
    semilogx : bool
        Whether plot data with semilogx or not.
    shape : tuple of ints, optional
        Original frame shape. For non-rectangular you must provide this so
        to define k step.
    size : int, optional
        If specified, perform log_averaging of data with provided size parameter.
        If not given, no averaging is performed.
    mask : ndarray, optional
        A boolean array indicating which data elements were computed.
    """
    fig = None
    data = None
    mask = None
    kmap = None
    anglemap = None
    indexmap = None
    k = 0
    angle = 0
    sector = 5
    kstep = 1
    mode = "real"
    
    def __init__(self, semilogx = True, shape = None, mask = None):
        self.semilogx = semilogx
        self.shape = shape
        self.computed_mask = mask
        if mask is not None:
            self.kisize, self.kjsize = mask.shape
        else:
            self.kisize, self.kjsize = None,None
               
    def _init_map(self):
        kisize, kjsize =  as_fft2_shape(self.kisize, self.kjsize, shape = self.shape)
        self.kmap, self.anglemap = fft2_kangle(kisize, kjsize, self.shape)
        self.indexmap = fft2_sector_indexmap(self.kmap, self.anglemap, self.angle, self.sector, self.kstep)

    def _init_graph(self):   
        self.graph = np.zeros(self.indexmap.shape)
        
        max_graph_value =self.kmap.max() #max(self.kmap[0,:].max(), self.kmap[:,0].max()) 
        self.graph[0,0] = max_graph_value 
        
        self._max_graph_value = max_graph_value        

    def _init_fig(self):
        if self.data is None:
            raise ValueError("You must first set data to plot!")
        self._init_map()
        self._init_graph()
        
        self.fig, (self.ax1,self.ax2) = plt.subplots(1,2,gridspec_kw = {'width_ratios':[3, 1]})
        
        self.fig.show()
        
        plt.subplots_adjust(bottom=0.25)


        self.im = self.ax2.imshow(np.fft.fftshift(self.graph), 
            extent=[0,self.graph.shape[1],self.graph.shape[0]//2+1,-self.graph.shape[0]//2-1])

        self.selectorax = plt.axes([0.1, 0.95, 0.65, 0.03])
        self.selectorindex = CustomRadioButtons(self.selectorax, ["real", "imag", "abs", "phase"], active = 0)
        
        
        self.kax = plt.axes([0.1, 0.15, 0.65, 0.03])
        self.kindex = Slider(self.kax, "k",0,int(self.kmap.max()),valinit = self.k, valfmt='%i')

        self.angleax = plt.axes([0.1, 0.10, 0.65, 0.03])
        self.angleindex = Slider(self.angleax, "angle",-90,90,valinit = self.angle, valfmt='%.2f')        

        self.sectorax = plt.axes([0.1, 0.05, 0.65, 0.03])
        self.sectorindex = Slider(self.sectorax, "sector",0,180,valinit = self.sector, valfmt='%.2f') 
        

        def update(val):
            self.set_mask(int(round(self.kindex.val)),self.angleindex.val,self.sectorindex.val, self.kstep)
            self.plot()        
        
        def update_mode(val):
            self.mode = val
            update(val)

            
        self.kindex.on_changed(update)
        self.angleindex.on_changed(update)
        self.sectorindex.on_changed(update)
        self.selectorindex.on_clicked(update_mode)
              
    def set_data(self, data, t = None):
        """Sets correlation data.
        
        Parameters
        ----------
        data : ndarray
            Normalized data.
        """
        self.kshape = data.shape[:-1]
        if self.computed_mask is None:
            self.kisize, self.kjsize = self.kshape
        self.data = data
        t =  np.asarray(t) if t is not None else np.arange(data.shape[-1])
        if len(t) != data.shape[-1]:
            raise ValueError("Wrong time array length")
        else:
            self.t = t
            
    def _get_avg_data(self):
        data = self.data[...,self.half_mask,:]
        m = self.conj_mask[self.half_mask]
        data[...,m,:] = np.conj(data[...,m,:])
        
        avg = np.nanmean(data, axis = -2)
        return self.t, data_repr(avg, self.mode)
    
    def get_normalized_data(self):
        if self.data is None:
            raise ValueError("No data, you must first set data.")
        return self.t, self.data
          
    def get_averaged_data(self):
        """Returns computed k-averaged data and time
        
        Returns
        -------
        x, y : ndarray, ndarray
            Time, data ndarrays.
        """
        
        if self.data is None:
            raise ValueError("No data, you must first set data.")
        if self.mask is None:
            raise ValueError("Mask not specified, you must first set mask.")
        
        return self._get_avg_data()
    
    def get_data(self):
        return self.get_averaged_data()
    
    
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
        self.indexmap = fft2_sector_indexmap(self.kmap, self.anglemap, self.angle, self.sector, self.kstep)
        self.mask = (self.indexmap == int(self.k)) 
        self.graph_mask = self.mask
        self.half_mask = as_rfft2_mask(self.mask)
        self.conj_mask = as_rfft2_conj(self.mask)
        
        if self.computed_mask is not None:
            if self.kshape == (self.kisize, self.kjsize):
                self.half_mask = self.half_mask & self.computed_mask
                self.graph_mask = self.mask & from_rfft2_data(self.computed_mask, self.shape)
                self.mask = self.graph_mask
                self.conj_mask = self.conj_mask & self.computed_mask
            else:
                self.graph_mask = self.mask & from_rfft2_data(self.computed_mask, self.shape)
                self.half_mask = self.half_mask[self.computed_mask]
                self.conj_mask = self.conj_mask[self.computed_mask]
                self.mask = self.mask[from_rfft2_data(self.computed_mask, self.shape)]
        
        return self.mask.any()
        
    def plot(self):
        """Plots data. You must first call :meth:`.set_data` to set input data"""
        
        if self.fig is None:
            self._init_fig()
            self.set_mask(int(round(self.k)),self.angle,self.sector, self.kstep)
            
        nans = (self.indexmap == -1)
        self.graph = self.indexmap*1.0 # make it float
        self.graph[self.graph_mask] = self._max_graph_value +1
        self.graph[nans] = np.nan
        
        self.im.set_data(np.fft.fftshift(self.graph))
                    
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

    def show(self,data = None):
        """Sets data and plots."""
        if data is not None:
            self.set_data(data)
        self.plot()
        
    def close(self):
        pass
        
    def __call__(self, i, data):
        return self.show(data = data)

class CorrViewer(DataViewer):
    """Plots raw correlation data. You need to hold reference to this object, 
    otherwise it will not work in interactive mode.

    Parameters
    ----------
    semilogx : bool
        Whether plot data with semilogx or not.
    shape : tuple of ints, optional
        Original frame shape. For non-rectangular you must provide this so
        to define k step.
    size : int, optional
        If specified, perform log_averaging of data with provided size parameter.
        If not given, no averaging is performed.
    norm : int, optional
        Normalization constant used in normalization
    scale : bool, optional
        Scale constant used in normalization.
    mask : ndarray, optional
        A boolean array indicating which data elements were computed.
    fold : bool
        For complex data this is a required parameter. I specifies whether to 
        fold data or not.
    """
    background = None
    variance = None
    
    def __init__(self, semilogx = True,  shape = None, size = None, norm = None, scale = False, mask = None, fold = None):
        self.norm  = norm
        self.scale = scale
        self.semilogx = semilogx
        self.shape = shape
        self.size = size
        self.computed_mask = mask
        self.fold = fold
        if mask is not None:
            self.kisize, self.kjsize = mask.shape
        
    def set_norm(self, value):
        """Sets norm parameter"""
        method = _method_from_data(self.data)
        self.norm = _default_norm_from_data(self.data,method,value)     
        
    def _init_fig(self):
        super()._init_fig()
        
        self.set_norm(self.norm)
        
        #self.rax = plt.axes([0.48, 0.55, 0.15, 0.3])
        self.cax = plt.axes([0.44, 0.72, 0.2, 0.15])
        
        self.active = [bool(self.norm & NORM_STRUCTURED),bool(self.norm & NORM_SUBTRACTED), bool((self.norm & NORM_WEIGHTED == NORM_WEIGHTED)) , bool((self.norm & NORM_COMPENSATED) == NORM_COMPENSATED)  ]
        
        self.check = CheckButtons(self.cax, ("structured", "subtracted", "weighted", "compensated"), self.active)
        
        #self.radio = RadioButtons(self.rax,("norm 0","norm 1","norm 2","norm 3","norm 4", "norm 5", "norm 6", "norm 7"), active = self.norm, activecolor = "gray")
 
        def update(label):
            index = ["structured", "subtracted", "weighted", "compensated"].index(label)
            status = self.check.get_status()

            norm = NORM_STRUCTURED if status[0] == True else NORM_STANDARD    
            
            if status[1]:
                norm = norm | NORM_SUBTRACTED
            if status[2]:
                norm = norm | NORM_WEIGHTED
            if status[3]:
                norm = norm | NORM_COMPENSATED
            try:
                self.set_norm(norm)
            except ValueError:
                self.check.set_active(index)
            self.set_mask(int(round(self.kindex.val)),self.angleindex.val,self.sectorindex.val, self.kstep)
            self.plot()

                
        self.check.on_clicked(update)

       
#        def update(val):
#            try:
#                self.set_norm(int(self.radio.value_selected[-1]))
#            except ValueError:
#                self.radio.set_active(self.norm)
#            self.set_mask(int(round(self.kindex.val)),self.angleindex.val,self.sectorindex.val, self.kstep)
#            self.plot()
#            
#        self.radio.on_clicked(update)

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
        self.kshape = data[0].shape[:-1]
        if self.computed_mask is None:
            self.kisize, self.kjsize = self.kshape
        self._normalized_data = None
            
    def get_normalized_data(self):
        if self.data is None:
            raise ValueError("No data, you must first set data.")
        if self._normalized_data is None:
            data = normalize(self.data, self.background, self.variance, norm = self.norm, scale = self.scale, mask = self.half_mask)
    
            if np.iscomplexobj(data) :
                if isinstance(self.background, tuple) or isinstance(self.variance, tuple) or self.fold is not None:
                    data = fold_data(data)
                elif self.background is None and self.variance is None:
                    raise ValueError("Cannot determine whether data folding is needed. You must explicitly define whether to fold correlation data or not.")
    
            if self.size is not None:
                # folding has already been made, so set to False
                t, data = log_average(data, self.size, fold = False)
            else:
                t = np.arange(data.shape[-1])   
            self._normalized_data = t, data
        return self._normalized_data
    
        
    def _get_avg_data(self):
        t, data = self.get_normalized_data()

        m = self.conj_mask[self.half_mask]
        data[...,m,:] = np.conj(data[...,m,:])
        
        avg = np.nanmean(data, axis = -2)
        
        return t, data_repr(avg, self.mode)    

    def __call__(self, i, data):
        if len(data) == 3:
            self.set_data(*data)
        else:
            self.set_data(data)
        return self.show()
    
            
class MultitauViewer(CorrViewer):
    """Shows multitau data in plot. You need to hold reference to this object, 
    otherwise it will not work in interactive mode.
    
    Parameters
    ----------
    semilogx : bool
        Whether plot data with semilogx or not.
    shape : tuple of ints, optional
        Original frame shape. For non-rectangular you must provide this so
        to define k step.
    norm : int, optional
        Normalization constant used in normalization
    scale : bool, optional
        Scale constant used in normalization.
    mask : ndarray, optional
        A boolean array indicating which data elements were computed.
    fold : bool
        For complex data this is a required parameter. I specifies whether to 
        fold data or not.
    """
    def __init__(self,  semilogx = True,  shape = None, norm = None, scale = False, mask = None, fold = None):

        self.norm  = norm
        self.scale = scale
        self.semilogx = semilogx
        self.shape = shape
        self.computed_mask = mask
        self.fold = fold
        if mask is not None:
            self.kisize, self.kjsize = mask.shape
        
    #@doc_inherit
    def set_norm(self, value):
        """Sets norm parameter"""
        method = _method_from_data(self.data[0])
        self.norm = _default_norm_from_data(self.data[0],method,value)  
        
    #@doc_inherit
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
        self.kshape = data[0][0].shape[:-1]
        if self.computed_mask is None:
            self.kisize, self.kjsize = self.kshape
        self._normalized_data = None
            
    def get_normalized_data(self):
        if self._normalized_data is None:
            data = normalize_multi(self.data, self.background, self.variance, norm = self.norm, scale = self.scale, mask = self.half_mask)
            
            if np.iscomplexobj(data[0]) :
                if isinstance(self.background, tuple) or isinstance(self.variance, tuple) or self.fold is not None:
                    data = fold_data(data[0],1), fold_data(data[1],1)
                elif self.background is None and self.variance is None:
                    raise ValueError("Cannot determine whether data folding is needed. You must explicitly define whether to fold correlation data or not.")
    
            if self.semilogx == True:
                t, data = log_merge(*data, fold = False)
            else:
                fast, slow = data
                t = np.arange(fast.shape[-1])
                data = fast
            self._normalized_data = t, data
        return self._normalized_data
        
    def _get_avg_data(self):
        t, data = self.get_normalized_data()
            
        m = self.conj_mask[self.half_mask]
        data[...,m,:] = np.conj(data[...,m,:])
        
        avg = np.nanmean(data, axis = -2)
        
        return t, data_repr(avg, self.mode)

        
#       
#if __name__ == "__main__":
#    video = (np.random.randn(256,256) for i in range(256))
#    vg = VideoViewer(video, 256, title = "iterator example") #must set nframes, because video has no __len__
#    vg.show()
#    
#    video = [np.random.randn(256,256) for i in range(256)] 
#    vl = VideoViewer(video, title = "list example") 
#    vl.show()  

    