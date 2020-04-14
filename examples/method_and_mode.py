"""
Demonstrates the use of method and mode options
"""

from cddm.viewer import DataViewer, CorrViewer
from cddm.video import multiply, normalize_video, crop, asarrays
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.core import acorr, normalize, stats, ccorr
from cddm.multitau import log_average
from cddm.multitau import iacorr_multi, normalize_multi, log_merge
from cddm.sim import simple_brownian_video, seed, numba_seed
import numpy as np
import matplotlib.pyplot as plt

from conf import SIZE, NFRAMES,DELTA


#set seeds so that all experiments are on ssame dataset
seed(0)
numba_seed(0)

#: this creates a brownian motion multi-frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame,)
video = simple_brownian_video(range(NFRAMES), shape = (SIZE+32,SIZE+32), delta = DELTA)

#: crop video to selected region of interest 
video = crop(video, roi = ((0,SIZE), (0,SIZE)))

#: create window for multiplication...
window = blackman((SIZE,SIZE))

#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES

#:perform the actual multiplication
video = multiply(video, window_video)

#: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
#video = normalize_video(video)

#: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
fft = rfft2(video, kimax =31, kjmax = 31)

#: you can also normalize each frame with respect to the [0,0] component of the fft
#: this it therefore equivalent to  normalize_video
#fft = normalize_fft(fft)

#load int numpy array
fft_array, = asarrays(fft, NFRAMES)

#: now perform auto correlation calculation with default parameters using iterative algorithm
acorr_data = acorr(fft_array, method = "fft")
corr, count, square_sum, data_sum, _ = acorr_data

ccorr_data = ccorr(fft_array, fft_array, method = "fft") #or method = "corr"
corr, count, square_sum, data_sum_1, data_sum_2 = ccorr_data

adiff_data = acorr(fft_array, method = "diff", norm = 1, n = 256)
diff, count, _, _ = adiff_data

cdiff_data = ccorr(fft_array, fft_array, method = "diff", norm = 3,  n = 256)
diff, count, data_sum1, data_sum2 = ccorr(fft_array, fft_array, method = "diff", norm = 3)

bg, var = stats(fft_array)

for data, method in zip((acorr_data, adiff_data),("corr","diff")):
    for mode in ("diff", "corr"):
        data_lin = normalize(data, bg, var, mode = mode, scale = True)
        plt.semilogx(data_lin[4,12], label = "mode = {}; method = {}".format(mode, method))

plt.legend()
plt.show()
    
