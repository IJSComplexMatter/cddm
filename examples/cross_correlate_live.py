"""
In this example we use a simulated dual-camera video of o brownian motion
of spherical particles and perform cross-ddm analysis.
"""

from cddm.sim import dual_frame_grabber, create_random_times2
from cddm.video import show_video, play, apply_window
from cddm.window import blackman
from cddm.fft import rfft2, show_fft 

from cddm import iccorr_multi, normalize, log_merge, k_select


import matplotlib.pyplot as plt

from cddm import conf
import numpy as np

#video iterator
from simple_brownian_video import get_dual_video
#trigger parameters
from simple_brownian_video import t1,t2, PERIOD, SHAPE


#setting this to 2 shows progress bar
conf.set_verbose(2)
#obtain frames iterator
dual_video = get_dual_video()
#apply blackman window 
#dual_video = apply_window(dual_video, blackman(SHAPE))
#dual_video = show_video(dual_video)
fdual_video = rfft2(dual_video, kisize = 64, kjsize = 64)
#fdual_video = show_fft(fdual_video)

#fdual_video = play(fdual_video,fps = 100)


data, bg = iccorr_multi(fdual_video, t1, t2, period = PERIOD, level = 5, 
                       chunk_size = 256*2, show = True, auto_background = False, binning =  True, return_background = True)
plt.figure()
plt.imshow(np.abs(bg[0]))


i,j = 4,8

plt.figure()

cfast, cslow = normalize(data)
x = np.arange(cfast.shape[-1])



#plot fast data  at k =(i,j) and for x > 0 (all but first data point)

plt.semilogx(x[1:], cfast[i,j][1:], "o", label = "fast")

#plot slow data
x = np.arange(cslow.shape[-1]) * PERIOD
for n, slow in enumerate(cslow):
    x = x * 2
    plt.semilogx(x[1:], slow[i,j][1:], "o", label = "slow {}".format(n+1))
    
#merged data
x, logdata = log_merge(cfast,cslow)
plt.semilogx(x[1:], logdata[i,j][1:], label = "merged")

plt.legend()

#np.save("ccorr_t.npy",x)
#np.save("ccorr_data.npy",logdata)


##now let us do some k-averaging
kdata = k_select(logdata, phi = 15, sector = 3, kstep = 1)

plt.figure()
#
for k, c in kdata: 
    print(k)
    plt.semilogx(x[1:], c[1:]/c[0], label = k)
plt.legend()
plt.show()



