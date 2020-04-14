"""Configuration parameters for the examples"""

import numpy as np

#: width and height of the frame
SIZE = 512
#: how many frames to simulate
NFRAMES = 1024
#: the delta parameter for the simulator
DELTA = 2
#: diffusion constant
D = 0.5*(DELTA*2*np.pi/SIZE)**2

KIMAX = 31
KJMAX = 31

KISIZE = KIMAX*2+1
KJSIZE = KJMAX+1
