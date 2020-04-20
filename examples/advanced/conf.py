"""Configuration parameters for the examples"""

import numpy as np
import cddm.conf

#: set verbose level
cddm.conf.set_verbose(2)

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
DELTA1 = 1
DELTA2 = 4
#: image static background value
BACKGROUND = 200
#: peak intensity of the particles
INTENSITY = 5
#: sigma of the gaussian of the particles
SIGMA = 3

MARGIN = 0

#computed parameters, do not change these
#----------------------------------------

SHAPE = (SIZE, SIZE)
SIMSHAPE = (SIZE + MARGIN, SIZE + MARGIN)

#size of the fft data
KISIZE = KIMAX*2+1
KJSIZE = KJMAX+1

#: diffusion constant
D1 = 0.5*(DELTA1*2*np.pi/SIZE)**2
D2 = 0.5*(DELTA2*2*np.pi/SIZE)**2

PERIOD = N_PARAMETER * 2
