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
NFRAMES = 1024
#: max k in the first axis
KIMAX = 39
#: max k in the second axis
KJMAX = 39
#: the delta parameter for the simulator
DELTA = 1
#: image static background value
BACKGROUND = 2**14
#: peak intensity of the particles
INTENSITY = 2**9
#: sigma of the gaussian of the particles
SIGMA = 3
#: simulation video margin size
MARGIN = 0
#: period of the random triggering
PERIOD = 8
#:how many particles to simulate
NUM_PARTICLES = 100

BIT_DEPTH = "12bit"

NOISE_MODEL = "gaussian"

READOUT_NOISE = 0.

SATURATION = 2**15

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

N_PARAMETER = PERIOD//2

NFRAMES_FULL = NFRAMES * int(PERIOD**0.5)


NFRAMES_RANDOM = NFRAMES

NFRAMES_STANDARD = NFRAMES*2

NFRAMES_DUAL = NFRAMES

#: clipping value of the image (max value)
VMAX = 2**int(BIT_DEPTH.split("bit")[0])
#: mean image value
VMEAN = BACKGROUND / SATURATION * VMAX




