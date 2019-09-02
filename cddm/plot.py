"""plotting functions and classes"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
#import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
#from matplotlib.patches import Circle

from ddm.core.ddm_tools import sector_indexmap, line_indexmap, correlate

from cdiff_new import log_merge, normalize_data

#I am using a class instead of function, so that plot objects have references and not garbage collected in interactive mode        
class show_video(object):
    """Shows video in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.
    """        
    def __init__(self,video, fftshift = False, logplot = False, cmap = None):
        
        def norm(frame):
            if fftshift:
                frame = np.fft.fftshift(frame)
            #if is complex or logplot
            if np.issubdtype(video.dtype, np.complexfloating) or logplot:
                frame = np.abs(frame)
                if logplot:
                    frame = np.log(1 + frame)
            return frame
                
        self.fig, self.ax1 = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        frame = norm(video[0])
        self.imax= self.ax1.imshow(frame,cmap = cmap)#, animated=True)
        
        #hist, n = np.histogram(frame.ravel(),255,(0,255))
        #self.hisl, = self.ax2.plot(hist)
        #self.cb = plt.colorbar()
    
        self.axindex = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.sindex = Slider(self.axindex, "frame",0,video.shape[0]-1,valinit = 0, valfmt='%i')
        
        def update(val):
            i = int(self.sindex.val)
            frame= norm(video[i])
            self.imax.set_data(frame)
            #hist, n = np.histogram(frame.ravel(),255,(0,255))
            #self.hisl.set_ydata(hist)
            self.fig.canvas.draw_idle()
        
        self.ids = self.sindex.on_changed(update)
        
#         self.playax = plt.axes([0.8, 0.025, 0.1, 0.04])
#         self.playbutton = Button(self.playax, 'Play', hovercolor='0.975')
#         self.stopax = plt.axes([0.6, 0.025, 0.1, 0.04])
#         self.stopbutton = Button(self.stopax, 'Stop', hovercolor='0.975')
# 
#         def update_anim(i):
#             self.imax.set_data(norm(video[i]))
#             self.sindex.set_val(i)
#             return self.imax,
#          
#         def play_video(event):
#              #disconnect slider move event
#             self.sindex.disconnect(self.ids)
#             if hasattr(self, "ani"):
#                 self.ani.event_source.stop()
#             start = int(self.sindex.val)
#             if start == video.shape[0] -1:
#                 start = 0
#             self.ani = FuncAnimation(self.fig, update_anim, frames=range(start,video.shape[0]), blit=True, interval = 1, repeat = False)
#             
#         def stop_video(event):
#             self.ani.event_source.stop()
#             del self.ani
#             gc.collect()
#             self.ids = self.sindex.on_changed(update)
#     
#         self.playbutton.on_clicked(play_video)
#         self.stopbutton.on_clicked(stop_video)    
        #plt.show()

def test_show_video():    
    video = np.random.randint(0,255,(1024,256,256),"uint8")
    plot = show_video(video)
    
class show_histogram(object):
    """Shows video in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.
    """        
    def __init__(self,video):
        self.fig, self.ax1 = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        frame = video[0]
        hist, n = np.histogram(frame.ravel(),255,(0,255))
        self.hisl, = self.ax1.plot(hist)
    
        self.axindex = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.sindex = Slider(self.axindex, "frame",0,video.shape[0]-1,valinit = 0, valfmt='%i')
        
        def update(val):
            i = int(self.sindex.val)
            frame= video[i]
            hist, n = np.histogram(frame.ravel(),255,(0,255))
            self.hisl.set_ydata(hist)
            self.fig.canvas.draw_idle()
        
        self.ids = self.sindex.on_changed(update)
        #plt.show()    
    
def test_show_histogram():    
    video = np.random.randint(0,255,(1024,256,256),"uint8")
    plot = show_histogram(video)
                
class show_fftdata(object):
    """Shows correlation data in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.
    """        
    def __init__(self,data, semilogx = True, normalize = True, sector = 5, angle = 0, kstep = 1):
        
        self.shape = data.shape
        self.data = data
                
        self.fig, (self.ax1,self.ax2) = plt.subplots(1,2,gridspec_kw = {'width_ratios':[3, 1]})
        plt.subplots_adjust(bottom=0.25)
        
        graph_shape = self.shape[0], self.shape[1]
        
        graph = np.zeros(graph_shape)
        
        max_k_value = np.sqrt((graph_shape[0]//2+1)**2+ graph_shape[1]**2) *np.sqrt(2)
        graph[0,0] = max_k_value 
        
        self.im = self.ax2.imshow(graph, extent=[0,self.shape[1],self.shape[0]//2+1,-self.shape[0]//2-1])
        #self.ax2.grid()
        
        norm = data[:,:,0]
        if normalize == False:
            norm = np.ones_like(norm)
        
        if semilogx == True:
            self.l, = self.ax1.semilogx(data[0,0,:]/norm[0,0])
        else:
            self.l, = self.ax1.plot(data[0,0,:]/norm[0,0])

        self.kax = plt.axes([0.1, 0.15, 0.65, 0.03])
        self.kindex = Slider(self.kax, "k",0,max_k_value,valinit = 0, valfmt='%i')

        self.phiax = plt.axes([0.1, 0.10, 0.65, 0.03])
        self.phiindex = Slider(self.phiax, "$\phi$",-90,90,valinit = angle, valfmt='%.2f')        

        self.sectorax = plt.axes([0.1, 0.05, 0.65, 0.03])
        self.sectorindex = Slider(self.sectorax, "sector",0,180,valinit = sector, valfmt='%.2f') 
                                  
        def update(val):
            k = self.kindex.val
            phi = self.phiindex.val
            sector = self.sectorindex.val
            ki = int(round(- k * np.sin(np.pi/180 * phi)))
            kj =  int(round(k * np.cos(np.pi/180 * phi)))
            #self.l.set_ydata(data[ki,kj,:]/norm[ki,kj])
            # recompute the ax.dataLim
            self.ax1.relim()

            # update ax.viewLim using the new dataLim
            self.ax1.autoscale_view()
            graph = np.zeros((self.shape[0], self.shape[1]))
            if sector != 0:
                indexmap = sector_indexmap(self.shape[0],self.shape[1],phi, sector, kstep)
            else:
                indexmap = line_indexmap(self.shape[0],self.shape[1],phi)
            #graph = np.concatenate((indexmap[:,:,np.newaxis],indexmap[:,:,np.newaxis],indexmap[:,:,np.newaxis]),-1) #make color image 
            graph = indexmap
            mask = (indexmap == int(round(k)))
            graph[mask] = max_k_value +1
            
            
            avg_data = correlate(self.data[mask,:]).mean(0)
            self.l.set_ydata(avg_data/avg_data[0])
            avg_graph = np.zeros(graph_shape)
            avg_graph[mask] = 1
            #graph[ki,kj] = 1
            #print (ki,kj)
            
            self.im.set_data(np.fft.fftshift(graph,0))
            
            #circ = Circle((ki,kj),5)
            #self.ax2.add_patch(circ)
            self.fig.canvas.draw_idle()  
        self.update = update      
                        
        self.kindex.on_changed(update)
        self.phiindex.on_changed(update)
        self.sectorindex.on_changed(update)
        plt.show()

class CorrelationViewer(object):
    """Shows correlation data in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.
    """        
    def init(self,data, semilogx = True, normalize = True):
        
        self.shape = data.shape
        self.data = data
                
        self.fig, (self.ax1,self.ax2) = plt.subplots(1,2,gridspec_kw = {'width_ratios':[3, 1]})
        plt.subplots_adjust(bottom=0.25)
        
        graph_shape = self.shape[0], self.shape[1]
        self._graph_shape = graph_shape
        
        graph = np.zeros(graph_shape)
        
        max_graph_value = max(graph_shape[0]//2+1, graph_shape[1]) 
        graph[0,0] = max_graph_value 
        
        self._max_graph_value = max_graph_value
        
        self.im = self.ax2.imshow(graph, extent=[0,self.shape[1],self.shape[0]//2+1,-self.shape[0]//2-1])
        #self.ax2.grid()
        
        norm = data[:,:,0]
        if normalize == False:
            norm = np.ones_like(norm)
        self._norm = norm
        self._data = data
        
        if semilogx == True:
            self.l, = self.ax1.semilogx(data[0,0,:]/norm[0,0])
        else:
            self.l, = self.ax1.plot(data[0,0,:]/norm[0,0])

        self.kax = plt.axes([0.1, 0.15, 0.65, 0.03])
        self.kindex = Slider(self.kax, "k",0,data.shape[1]-1,valinit = 0, valfmt='%i')

        self.phiax = plt.axes([0.1, 0.10, 0.65, 0.03])
        self.phiindex = Slider(self.phiax, "$\phi$",-90,90,valinit = 0, valfmt='%.2f')        

        self.sectorax = plt.axes([0.1, 0.05, 0.65, 0.03])
        self.sectorindex = Slider(self.sectorax, "sector",0,180,valinit = 5, valfmt='%.2f') 
                                  
        def update(val):
            self.update()
            
        self.kindex.on_changed(update)
        self.phiindex.on_changed(update)
        self.sectorindex.on_changed(update)
        
        
        def update2(*args):
            self.update()
            return self.l,

        self.ani = FuncAnimation(self.fig, update2, interval=1000, blit=False)
        
        
    def update(self):
        k = self.kindex.val
        phi = self.phiindex.val
        sector = self.sectorindex.val
        ki = int(round(- k * np.sin(np.pi/180 * phi)))
        kj =  int(round(k * np.cos(np.pi/180 * phi)))
        self.l.set_ydata(self._data[ki,kj,:]/self._norm[ki,kj])

        graph = np.zeros((self.shape[0], self.shape[1]))
        if sector != 0:
            indexmap = sector_indexmap(self.shape[0],self.shape[1],phi, sector, 1)
        else:
            indexmap = line_indexmap(self.shape[0],self.shape[1],phi)
        #graph = np.concatenate((indexmap[:,:,np.newaxis],indexmap[:,:,np.newaxis],indexmap[:,:,np.newaxis]),-1) #make color image 
        graph = indexmap
        mask = (indexmap == int(round(k)))
        graph[mask] = self._max_graph_value +1
        
        
        avg_data = self.data[mask,:].mean(0)
        self.l.set_ydata(avg_data/avg_data[0])
        
        # recompute the ax.dataLim
        self.ax1.relim()

        # update ax.viewLim using the new dataLim
        self.ax1.autoscale_view()
        
        
        avg_graph = np.zeros(self._graph_shape)
        avg_graph[mask] = 1
        #graph[ki,kj] = 1
        #print (ki,kj)
        
        self.im.set_data(np.fft.fftshift(graph,0))
        
        #circ = Circle((ki,kj),5)
        #self.ax2.add_patch(circ)
        #self.fig.canvas.draw_idle() 
        #self.fig.canvas.flush_events()
        

        #try:
        #    self.fig.canvas.start_event_loop(0.5)
        #except:
        #    pass
        
    def show(self):
        """Shows video."""
        self.fig.show()

def test_show_correlation():    
    video = np.random.randint(0,255,(256,129,1024*8),"uint8")
    viewer = CorrelationViewer()
    viewer.init(video)
    return viewer
    

class CorrelationViewer2(object):
    """Shows correlation data in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.
    """        
    def init(self,data, semilogx = True, normalize = True):
        
        #self.shape = data.shape
        (self.count_fast, self.data_fast), (self.count_slow, self.data_slow) = data
        self.shape = self.data_fast.shape[0:-1]
        
        self.fig, (self.ax1,self.ax2) = plt.subplots(1,2,gridspec_kw = {'width_ratios':[3, 1]})
        
        self.fig.show()
        plt.subplots_adjust(bottom=0.25)
        
        graph_shape = self.shape[0], self.shape[1]
        self._graph_shape = graph_shape
        
        graph = np.zeros(graph_shape)
        
        max_graph_value = max(graph_shape[0]//2+1, graph_shape[1]) 
        graph[0,0] = max_graph_value 
        
        self._max_graph_value = max_graph_value
        
        self.im = self.ax2.imshow(graph, extent=[0,self.shape[1],self.shape[0]//2+1,-self.shape[0]//2-1])
        #self.ax2.grid()
        
        cfast_avg = self.data_fast[0,0]
        cslow_avg = self.data_slow[:,0,0]
        
        normalize_data(cfast_avg, self.count_fast, cfast_avg)
        normalize_data(cslow_avg, self.count_slow, cslow_avg)
        
        t, avg_data = log_merge(cfast_avg,cslow_avg)
        
        
        if semilogx == True:
            self.l, = self.ax1.semilogx(avg_data/avg_data[0])
        else:
            self.l, = self.ax1.plot(avg_data/avg_data[0])

        self.kax = plt.axes([0.1, 0.15, 0.65, 0.03])
        self.kindex = Slider(self.kax, "k",0,self.shape[1]-1,valinit = 0, valfmt='%i')

        self.phiax = plt.axes([0.1, 0.10, 0.65, 0.03])
        self.phiindex = Slider(self.phiax, "$\phi$",-90,90,valinit = 0, valfmt='%.2f')        

        self.sectorax = plt.axes([0.1, 0.05, 0.65, 0.03])
        self.sectorindex = Slider(self.sectorax, "sector",0,180,valinit = 5, valfmt='%.2f') 
                                  
        def update(val):
            self.update()
            
        self.kindex.on_changed(update)
        self.phiindex.on_changed(update)
        self.sectorindex.on_changed(update)
        
        
        def update2(*args):
            self.update()
            return self.l,

        #self.ani = FuncAnimation(self.fig, update2, interval=1000, blit=False)
        
        
    def update(self):
        k = self.kindex.val
        phi = self.phiindex.val
        sector = self.sectorindex.val
        #ki = int(round(- k * np.sin(np.pi/180 * phi)))
        #kj =  int(round(k * np.cos(np.pi/180 * phi)))
        #self.l.set_ydata(self._data[ki,kj,:]/self._norm[ki,kj])

        graph = np.zeros((self.shape[0], self.shape[1]))
        if sector != 0:
            indexmap = sector_indexmap(self.shape[0],self.shape[1],phi, sector, 1)
        else:
            indexmap = line_indexmap(self.shape[0],self.shape[1],phi)
        #graph = np.concatenate((indexmap[:,:,np.newaxis],indexmap[:,:,np.newaxis],indexmap[:,:,np.newaxis]),-1) #make color image 
        graph = indexmap
        mask = (indexmap == int(round(k)))
        graph[mask] = self._max_graph_value +1
        

        
        cfast_avg = self.data_fast[mask,:].mean(0)
        cslow_avg = self.data_slow[:,mask,:].mean(1)
        
        normalize_data(cfast_avg, self.count_fast, cfast_avg)
        normalize_data(cslow_avg, self.count_slow, cslow_avg)
        
        t, avg_data = log_merge(cfast_avg,cslow_avg)
        

        self.l.set_ydata(avg_data/avg_data[0])
        self.l.set_xdata(t)
        
        # recompute the ax.dataLim
        self.ax1.relim()

        # update ax.viewLim using the new dataLim
        self.ax1.autoscale_view()
        
        
        avg_graph = np.zeros(self._graph_shape)
        avg_graph[mask] = 1
        #graph[ki,kj] = 1
        #print (ki,kj)
        
        self.im.set_data(np.fft.fftshift(graph,0))
        
        #circ = Circle((ki,kj),5)
        #self.ax2.add_patch(circ)
        self.fig.canvas.draw() 
        self.fig.canvas.flush_events()
        
        plt.pause(0.01)
        

        #try:
        #    self.fig.canvas.start_event_loop(0.5)
        #except:
        #    pass
        
    def show(self):
        """Shows video."""
        self.fig.show()
    
    
