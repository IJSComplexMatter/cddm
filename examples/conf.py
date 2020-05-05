"""Configuration parameters for the examples"""
import numpy as np
import cddm.conf

#: set verbose level
#cddm.conf.set_verbose(2)

#settable parameters, you can change these 
#-----------------------------------------

#: width and height of the frame
SIZE = 512
#: how many frames to simulate
NFRAMES = 1024
#: max k in the first axis
KIMAX = 31
#: max k in the second axis
KJMAX = 31
#: n parameter of the random triggering 
N_PARAMETER = 16
#: the delta parameter for the simulator
DELTA = 2
#: image static background value
BACKGROUND = 200
#: peak intensity of the particles
INTENSITY = 5
#: sigma of the gaussian of the particles
SIGMA = 3
#: simulation video margin size
MARGIN = 32

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
#: effective period of the dual camera experiment
PERIOD = N_PARAMETER * 2

import os.path as p

#: data paths
PATH = p.split(p.abspath(__file__))[0]
DATA_PATH = p.join(PATH, "data")
DUST1_PATH = p.join(DATA_PATH, "dust1.png")
DUST2_PATH = p.join(DATA_PATH, "dust2.png")
