"""Configuration parameters for the examples"""
import numpy as np
import cddm.conf

from examples.conf import DATA_PATH, DUST1_PATH, DUST2_PATH

#: set verbose level
cddm.conf.set_verbose(2)

#settable parameters, you can change these 
#-----------------------------------------

#: width and height of the frame
SIZE = 512
#: how many frames to simulate
NFRAMES = 1024*4
#: max k in the first axis
KIMAX = 31
#: max k in the second axis
KJMAX = 31
#: the delta parameter for the simulator
DELTA = 1
#: image static background value
BACKGROUND = 16384
#: peak intensity of the particles
INTENSITY = 512
#: sigma of the gaussian of the particles
SIGMA = 3
#: simulation video margin size
MARGIN = 32
#: period of the random triggering
PERIOD = 32

NFRAMES_FULL = NFRAMES * int(PERIOD**0.5)

#computed parameters, do not change these
#----------------------------------------

#: shape of the cropped video
SHAPE = (SIZE, SIZE)
#: shape of the simulation video
SIMSHAPE = (SIZE + MARGIN, SIZE + MARGIN)

#: size of the fft data
KISIZE = KIMAX*2+1
KJSIZE = KJMAX+1

#: diffusion constant
D = 0.5*(DELTA*2*np.pi/SIZE)**2



