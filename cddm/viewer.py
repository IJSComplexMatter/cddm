"""A simple matlotlib-based video viewer. Video can be a list-like object or an
iterator that yields 2D array."""
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class VideoViewer(object):
    """
    A matplotlib-based video viewer.
    
    Parameters
    ----------
    video : list-like, iterator
        A list of a tuple of 2D arrays or a generator of a tuple of 2D arrays. 
        If an iterator is provided, you must set 'n' as well. 
    n: int
        Length of the video. When this is set it displays only first 'n' frames of the video.
    id : int, optional
        For multi-frame data specifies camera index.
    title : str, optional
        Plot title.
    """

    def __init__(self, video, n = None, id = 0, title = ""):

        if n is None:
            try:
                n = len(video)
            except TypeError:
                raise Exception("You must specify nframes!")
        self.id = id
        self.index = 0
        self.video = video
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        plt.subplots_adjust(bottom=0.25)
        
        frame = next(iter(video)) #take first frame
        frame = self._prepare_image(frame)
        self.img = self.ax.imshow(frame) 

        self.axframe= plt.axes([0.25, 0.1, 0.65, 0.03])
        self.sframe = Slider(self.axframe, 'Frame', 0, n - 1, valinit=0, valstep=1, valfmt='%i')
        
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
        
    def _prepare_image(self,im):
        if isinstance(im, tuple) or isinstance(im, list):
            return im[self.id]
        else:
            return im
        
    def show(self):
        """Shows video."""
        plt.show()
        
if __name__ == "__main__":
    video = (np.random.randn(256,256) for i in range(256))
    vg = VideoViewer(video, 256, title = "iterator example") #must set nframes, because video has no __len__
    vg.show()
    
    video = [np.random.randn(256,256) for i in range(256)] 
    vl = VideoViewer(video, title = "list example") 
    vl.show()   
    