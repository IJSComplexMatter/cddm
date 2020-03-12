#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script simulates brownian motion of 100 particles. A normal DDM video,
and a cross-ddm video (two videos) are writen to disk.
(about 3 x 1 GB of data written)
"""

from cddm import viewer
from cddm import sim 
from cddm import video
from cddm import conf
import numpy as np

conf.set_verbose(2)

n = 16
PERIOD = n*2
NFRAMES_DUAL = 1024*4
NFRAMES_SINGLE = NFRAMES_DUAL #// 2 // n

t1, t2 = sim.create_random_times2(nframes = NFRAMES_DUAL, n = n)


MARGIN = 32
SHAPE = (512,512)
#final shape will be 512x512, but here we add some border area so that after
#cropping the image to 512x512 this removes mirroring condition at the borders of
#the simulation.
SIMSHAPE = SHAPE[0] + 2* MARGIN, SHAPE[1] + 2* MARGIN
SIMFRAMES = max(max(t1), max(t2)) + 1
NPARTICLES = 100

#def _add_frames_and_crop(frames1, frames2):
#    """adds two frames dogether from a multi-frame input data nad crops results"""
#    return tuple(((frame1+frame2)[MARGIN:SIMSHAPE[0]-MARGIN,0:SHAPE[1]] for frame1, frame2 in zip(frames1, frames2))) 

def _crop_frames(frames):
    return tuple((frame[0:SHAPE[0],0:SHAPE[1]] for frame in frames))

def intensity_jitter(vid, t1,t2):
    import scipy.misc
    face = scipy.misc.face(gray = True)[0:SIMSHAPE[0],0:SIMSHAPE[1]]
    facem = scipy.misc.face(gray = True)[-SIMSHAPE[0]-1:-1,-SIMSHAPE[1]-1:-1]
    maxdt = max(np.abs(t1-t2))
    for i, frames in enumerate(vid):
        #frames = frames[0]+ face, frames[1]+facem
        if t1[i] == t2[i]:
            yield frames
        elif t1[i] > t2[i]:
            #yield frames
            #print(frames[0].dtype)
            yield np.uint8(frames[0] * (1- 0.5/maxdt*(maxdt - np.abs(t1[i]-t2[i])))), frames[1]
        else:
            yield frames[0], np.uint8(frames[1] * (1- 0.5/maxdt*(maxdt - np.abs(t1[i]-t2[i]))))
def get_dual_video(seed = None):
    
    #We use a mix of numba and numpy random generator... 
    #If you want to seed, both need to be set!, this seed  function can be used for that
    if seed is not None:
        sim.seed(seed)
        
    #build particle coordinates  of smaller, faster particles
    x1 = sim.brownian_particles(shape = SIMSHAPE, n = SIMFRAMES, delta = 1., dt = 1, particles = NPARTICLES)   
    #build particle coordinate of a single large slow moving particle 
    #x2 = sim.brownian_particles(shape = SIMSHAPE, n = SIMFRAMES, delta = 0.1, dt = 1,
    #                           particles = 1, velocity = ((0.01,0),), x0 = ((0,256),)) 
    
    dual_vid = sim.particles_video(x1, t1 = t1, t2 = t2, shape = SIMSHAPE, sigma = 5, intensity = 5, background = 0)
    #dual_vid2 = sim.particles_video(x2,t1 = t1, t2 = t2,  shape = SIMSHAPE, sigma = 30, intensity = 30, background = 0)
    #dual_vid = intensity_jitter(dual_vid, t1,t2)
    return (_crop_frames(frames) for frames in dual_vid)

def get_video(n = NFRAMES_SINGLE, seed = None):
    
    #We use a mix of numba and numpy random generator... 
    #If you want to seed, both need to be set!, this seed  function can be used for that
    if seed is not None:
        sim.seed(seed)
        
    #build particle coordinates  of smaller, faster particles
    x1 = sim.brownian_particles(shape = SIMSHAPE, n = NFRAMES_SINGLE, delta = 1., dt = 1, particles = NPARTICLES)   
    #build particle coordinate of a single large slow moving particle 
    #x2 = sim.brownian_particles(shape = SIMSHAPE, n = NFRAMES_SINGLE, delta = 0.1, dt = 1,
    #                           particles = 1, velocity = ((0.01,0),), x0 = ((0,256),)) 
    
    vid1 = sim.particles_video(x1, shape = SIMSHAPE, sigma = 5, intensity = 10, background = 0)
    #vid2 = sim.particles_video(x2,  shape = SIMSHAPE, sigma = 30, intensity = 30, background = 0)

    return (_crop_frames((frame,)) for frame in vid1)
    #return ((frame1+frame2,frame1+frame2) for frame1,frame2 in zip(vid1,vid2))


if __name__ == "__main__":

    print("Computing ddm video...")
    vid = get_video(seed = 0)
    vid, = video.asarrays(vid,NFRAMES_SINGLE)
    print("Writing to HD ...")
    np.save("simple_brownian_ddm_video.npy",vid)
    
    print("Computing cddm video...")
    vid = get_dual_video(seed = 0)
    
    
    #we can write directly to HD by creating and writing to numpy memmap
    v1,v2 = video.asmemmaps("simple_brownian_cddm_video", vid, NFRAMES_DUAL)
    
    np.save("simple_brownian_cddm_t1.npy", t1)
    np.save("simple_brownian_cddm_t2.npy", t2)
    #print("Writing to HD ...")
    #np.save("brownian_dual_camera_0.npy",v1)
    #np.save("brownian_dual_camera_1.npy",v2)
   
    #v1 = viewer.VideoViewer(get_dual_video(), n = 1024)
    #v1.show()

    v = viewer.VideoViewer(v1)
    v.show()

    v = viewer.VideoViewer(v2)
    v.show()
