"""Additional configuration parameters for the one_component simulations"""
import numpy as np
from examples.paper.conf import *

#settable parameters, you can change these 
#-----------------------------------------

#: peak intensity of the particles
INTENSITY = 2**9
#: sigma of the gaussian PSF of the particles
SIGMA = 3
#: how many particles are in the box
NUM_PARTICLES = 100
#: the delta parameter for the simulator
DELTA = 1

#computed parameters, do not change these
#----------------------------------------

#: diffusion constant
D = 0.5*(DELTA*2*np.pi/SIZE)**2






