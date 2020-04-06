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

import matplotlib.pyplot as plt

#: whether to apply dust particles to video
APPLY_DUST = True

#frame parameters
SIZE = 512
SHAPE = (SIZE,SIZE)

DELTA = 2
#diffusion constant
D = 2*(DELTA*np.pi/SIZE)**2

dust1 = plt.imread('dust1.png')[...,0] #float normalized to (0,1)
dust2 = plt.imread('dust2.png')[...,0]
DUST = dust1, dust2

conf.set_verbose(2)

#the n parameter (from the CDDM paper) in dual video.
n = 16
#actual period is two times of that
PERIOD = n*2

#how many frames to capture
NFRAMES_DUAL = 1024*4
NFRAMES_SINGLE = NFRAMES_DUAL #// 2 // n

sim.seed(0)
#build random time sequence for cross DDM
t1, t2 = sim.create_random_times2(nframes = NFRAMES_DUAL, n = n)

#extra margin, to avoid mirror boundary conditions.. these margins are cropped
MARGIN = 32

#final shape will be 512x512, but here we add some border area so that after
#cropping the image to 512x512 this removes mirroring condition at the borders of
#the simulation.
SIMSHAPE = SHAPE[0] + 2* MARGIN, SHAPE[1] + 2* MARGIN

#how many frames must we simulate
SIMFRAMES = max(max(t1), max(t2)) + 1
#: number of particles in the frame.
NPARTICLES = 100


def _crop_frames(frames):
    """Crops frames and applies dust"""
    if APPLY_DUST == True:
        return tuple((np.uint8(frame[0:SHAPE[0],0:SHAPE[1]] * DUST[i]) for i,frame in enumerate(frames)))
    else:
        return tuple((np.uint8(frame[0:SHAPE[0],0:SHAPE[1]] ) for i,frame in enumerate(frames)))

def get_dual_video(seed = None):
    
    #We use a mix of numba and numpy random generator... 
    #If you want to seed, both need to be set!, this seed  function can be used for that
    if seed is not None:
        sim.seed(seed)
        
    #build particle coordinates  
    x1 = sim.brownian_particles(shape = SIMSHAPE, n = SIMFRAMES, delta = DELTA, dt = 1, particles = NPARTICLES)   

    video = sim.particles_video(x1, t1 = t1, t2 = t2, shape = SIMSHAPE, sigma = 5, intensity = 5, background = 200)

    return (_crop_frames(frames) for frames in video)

def get_video(n = NFRAMES_SINGLE, seed = None):
    
    #We use a mix of numba and numpy random generator... 
    #If you want to seed, both need to be set!, this seed  function can be used for that
    if seed is not None:
        sim.seed(seed)
        
    x1 = sim.brownian_particles(shape = SIMSHAPE, n = NFRAMES_SINGLE, delta = DELTA, dt = 1,  particles = NPARTICLES)   

    video = sim.particles_video(x1, shape = SIMSHAPE, sigma = 5, intensity = 5, background = 200)
    
    return (_crop_frames((frame,)) for frame in video)


if __name__ == "__main__":

    print("Computing ddm video...")
    vid = get_video(seed = 0)
    vid, = video.asarrays(vid,NFRAMES_SINGLE)
    print("Writing to HD ...")
    np.save("simple_brownian_ddm_video.npy",vid)
    
    print("Computing cddm video...")
    vid = get_dual_video(seed = 0)
    
    v1, v2 = video.asarrays(vid,NFRAMES_DUAL)
    
    print("Writing to HD ...")
    np.save("brownian_dual_camera_0.npy",v1)
    np.save("brownian_dual_camera_1.npy",v2)
    
#    #we can write directly to HD by creating and writing to numpy memmap
#    vid = get_dual_video(seed = 0)
#    v1,v2 = video.asmemmaps("simple_brownian_cddm_video", vid, NFRAMES_DUAL)
#    
    np.save("simple_brownian_cddm_t1.npy", t1)
    np.save("simple_brownian_cddm_t2.npy", t2)

   
    v = viewer.VideoViewer(v1, title = "camera 1")
    v.show()

    v = viewer.VideoViewer(v2, title = "camera 2")
    v.show()
