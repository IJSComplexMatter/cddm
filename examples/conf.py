"""Configuration parameters for the examples"""

import numpy as np

#settable parameters, you can change these
#-----------------------------------------

#: width and height of the frame
SIZE = 512
#: how many frames to simulate
NFRAMES = 1024
#: the delta parameter for the simulator
DELTA = 2
#: max k in the first axis
KIMAX = 31
#: max k in the second axis
KJMAX = 31


#computed parameters, do not change these
#----------------------------------------

#size of the fft data
KISIZE = KIMAX*2+1
KJSIZE = KJMAX+1

#: diffusion constant
D = 0.5*(DELTA*2*np.pi/SIZE)**2
