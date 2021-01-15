"""Additional configuration parameters for the one_component simulations"""
import numpy as np
from examples.paper.conf import *

#settable parameters, you can change these 
#-----------------------------------------

#: peak intensity of the particles
INTENSITY1 = 2**9
INTENSITY2 = 2**9

#: sigma of the gaussian PSF of the particles
SIGMA1 = 3
SIGMA2 = 3
#: how many particles are in the box
NUM_PARTICLES1 = 20
NUM_PARTICLES2 = 100
#: the delta parameter for the simulator
DELTA1 = 0.5
DELTA2 = 4

#computed parameters, do not change these
#----------------------------------------

#: diffusion constant
D1 = 0.5*(DELTA1*2*np.pi/SIZE)**2

#: diffusion constant
D2 = 0.5*(DELTA2*2*np.pi/SIZE)**2




