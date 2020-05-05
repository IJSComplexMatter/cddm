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
KIMAX = 41
#: max k in the second axis
KJMAX = 41
#: n parameter of the random triggering 
N_PARAMETER = 16
#: the delta parameter for the simulator
DELTA1 = 1.5
DELTA2 = 4.
#: image static background value
BACKGROUND = 200
#: peak intensity of the particles
INTENSITY1 = 5
INTENSITY2 = 5
#: sigma of the gaussian of the particles
SIGMA1 = 5
SIGMA2 = 3
#: mean velocity of particles
VELOCITY = (0.0,0.0)

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
