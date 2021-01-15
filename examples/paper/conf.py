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
KIMAX = 51
#: max k in the second axis
KJMAX = 51
#: image static background value
BACKGROUND = 2**14
#: simulation video margin size
MARGIN = 32
#: period of the random triggering
PERIOD = 64
#: bit depth of ADC converter
BIT_DEPTH = "12bit"
#: noise model for shot noise (either 'poisson' or 'gaussian' or 'none' if you want to disable photon noise)
NOISE_MODEL = "gaussian"
#: extra noise added to the image (in addition to photon noise)
READOUT_NOISE = 0.
#: sensor saturation value above which data is clipped
SATURATION = 2**15
#: enable or disable FFT windowing
APPLY_WINDOW = True
#: whether to apply dust particles or not
APPLY_DUST = True
#: whether to save plots, or show only.
SAVE_FIGS = False

if SAVE_FIGS:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,     # use inline math for ticks
        #"pgf.rcfonts": False,    # don't setup fonts from rc parameters
        # "pgf.preamble": "\n".join([
        #      "\\usepackage{units}",          # load additional packages
        #      "\\usepackage{metalogo}",
        #      "\\usepackage{unicode-math}",   # unicode math setup
        #      r"\setmathfont{xits-math.otf}",
        #      r"\setmainfont{DejaVu Serif}",  # serif font via preamble
        # ])
    })

    # plt.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     "pgf.preamble": "\n".join([
    #          r"\usepackage[utf8x]{inputenc}",
    #          r"\usepackage[T1]{fontenc}",
    #          r"\usepackage{cmbright}",
    #     ]),
    # })
    
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     'font.size' : 11,
    #     'font.family' : 'lmodern',
    #     #'text.latex.unicode': True,
    #     #"font.family": "sans-serif",
    #     #"font.sans-serif": ["Helvetica"]
    #     })
    # ## for Palatino and other serif fonts use:
    # #plt.rcParams.update({
    # #    "text.usetex": True,
    # #    "font.family": "serif",
    # #    "font.serif": ["Palatino"],
    # #})


#computed parameters, do not change these
#----------------------------------------

#: shape of the cropped video
SHAPE = (SIZE, SIZE)
#: shape of the simulation video
SIMSHAPE = (SIZE + MARGIN, SIZE + MARGIN)

#: size of the fft data
KISIZE = KIMAX*2+1
KJSIZE = KJMAX+1

PERIOD_RANDOM = PERIOD // 2

#: n parameter for irregular triggering
N_PARAMETER = PERIOD//2


NFRAMES_RANDOM = NFRAMES * 2

NFRAMES_STANDARD = NFRAMES * 2

NFRAMES_FULL = NFRAMES_STANDARD *4

NFRAMES_DUAL = NFRAMES

NFRAMES_FAST = NFRAMES * 2

DT_DUAL = 1

DT_STANDARD = DT_DUAL* PERIOD//2

DT_RANDOM =DT_DUAL

DT_FAST = DT_DUAL

DT_FULL = DT_STANDARD // 4

#: clipping value of the image (max value)
VMAX = 2**int(BIT_DEPTH.split("bit")[0])
#: mean image value
VMEAN = BACKGROUND / SATURATION * VMAX
#: viewing area ratio factor
AREA_RATIO = SHAPE[0]*SHAPE[1]/(SIMSHAPE[0] * SIMSHAPE[1]) 
#: adc scalling factor
ADC_SCALE_FACTOR = SATURATION / VMAX






