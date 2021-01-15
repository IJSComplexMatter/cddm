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
NFRAMES = 1024*4*4
#: max k in the first axis
KIMAX = 41
#: max k in the second axis
KJMAX = 41
#: n parameter of the random triggering 
N_PARAMETER = 16
#: the delta parameter for the simulator
DELTA1 = 0.5
DELTA2 = 2
#: image static background value must be less than 2**16
BACKGROUND = 2**14
#: sensort saturation value, must be less than 2**16
SATURATION = 2**15
#: ADC bit depth
ADC_BIT_DEPTH = "10bit"
#: readout noise level
READOUT_NOISE = 0
#: noise model for readout noise and shot noise
NOISE_MODEL = "poisson"
#: peak intensity of the particles
INTENSITY1 = 2**10
INTENSITY2 = 2**9
#: sigma of the gaussian of the particles
SIGMA1 = 3
SIGMA2 = 5
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
