"""Computes and plots form factor. (Fig 2. in the paper)"""

# from cddm.viewer import MultitauViewer

from cddm.window import blackman
from cddm.sim import form_factor
from cddm.norm import noise_level
from cddm.fft import rfft2_crop

import matplotlib.pyplot as plt

import numpy as np
from examples.paper.conf import SHAPE, BACKGROUND, APPLY_WINDOW, AREA_RATIO,  ADC_SCALE_FACTOR, KIMAX, KJMAX, SAVE_FIGS

from examples.paper.simple_video.conf import D, SIGMA, INTENSITY, NUM_PARTICLES, VMEAN, DUST1_PATH, DUST2_PATH, APPLY_DUST
from cddm.norm import  weight_prime_from_g, weight_from_g

#: create window for multiplication...
window = blackman(SHAPE)

if APPLY_WINDOW == False:
    #we still need window for form_factor calculation
    window[...] = 1.

print("Computing form factor")
#form factor, for relative signal intensity calculation
formf = form_factor(window, sigma = SIGMA, intensity = INTENSITY, dtype = "uint16", navg = 30, mode = "fft2")

ifreq = np.fft.fftfreq(SHAPE[0], d=1/SHAPE[0])
jfreq = np.fft.fftfreq(SHAPE[1], d=1/SHAPE[1])

K_RADIUS = 35

if APPLY_DUST:
    dust1 = plt.imread(DUST1_PATH)[0:SHAPE[0],0:SHAPE[1],0] #float normalized to (0,1)
    dust2 = plt.imread(DUST2_PATH)[0:SHAPE[0],0:SHAPE[1],0] #float normalized to (0,1)


def g1(x, kimax = None, kjmax = None, cross = True):
    i,j=np.meshgrid(ifreq,jfreq, indexing = "ij")
    
    x = np.asarray(x)
    
    #add dimensions for broadcasting
    x = x[..., None,None]

    #expected signal
    a = NUM_PARTICLES * formf **2 * AREA_RATIO
    #expected variance (signal + noise)
    if APPLY_DUST:
        if cross:
            v = a + noise_level((window*dust1+window*dust2)/2,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
        else:
            v = a + noise_level(window*dust1,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5

    else:
        v = a + noise_level(window,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5

    #expected scalling factor
    a = a/v   
    
    a = rfft2_crop(a, kimax,kjmax)
    i = rfft2_crop(i, kimax,kjmax)
    j = rfft2_crop(j, kimax,kjmax)
    
    return a * np.exp(-D*(i**2+j**2)*x)
    
def fvar(kimax = None, kjmax = None, cross = True):
    #expected signal
    a = NUM_PARTICLES * formf**2 * AREA_RATIO
    if APPLY_DUST:
        if cross:
            v = a + noise_level((window*dust1+window*dust2)/2,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
        else:
            v = a + noise_level(window*dust1,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5        
    else:
        v = a + noise_level(window,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5

    v = v/(ADC_SCALE_FACTOR**2)
    return rfft2_crop(v, kimax,kjmax)

def bg1(kimax = None, kjmax = None):
    if APPLY_DUST:
        a = (np.fft.fft2(dust1*window)*VMEAN) / (fvar())**0.5
    else:
        a = (np.fft.fft2(window)*VMEAN) / (fvar())**0.5
    return rfft2_crop(a, kimax,kjmax)
    

def bg2(kimax = None, kjmax = None):
    if APPLY_DUST:
        a = (np.fft.fft2(dust2*window)*VMEAN) / (fvar())**0.5
    else:
        a = (np.fft.fft2(window)*VMEAN) / (fvar())**0.5
    return rfft2_crop(a, kimax,kjmax)
        
def delta():
    return 0.

def weights(x, kimax = None, kjmax = None, cross = True):
    """Returns optimal weight and weight prime."""

    _bg1 = bg1(kimax,kjmax)
    _bg2 = bg2(kimax,kjmax)
    delta = 0.
    
    _g1 = g1(x, kimax, kjmax, cross)
    
    w = weight_from_g(_g1,delta)
    wp = weight_prime_from_g(_g1,delta, _bg1,_bg2)
    
    return w, wp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
  
    fig = plt.figure(figsize = (6.4,2.8))
    linewidth=72/fig.dpi

    ax = plt.subplot(121)
    
    ii = np.fft.fftshift(ifreq)
    jj = np.fft.fftshift(jfreq)
    rect = patches.Rectangle((-KIMAX,-KJMAX),KJMAX*2+1,KIMAX*2+1,linewidth=linewidth,edgecolor='w',facecolor='none')
    patch= patches.Circle((0,0),35,linewidth=linewidth,edgecolor='r',facecolor='none')
    
    plt.imshow(np.fft.fftshift(formf **2),extent=(ii[0],ii[-1],jj[0],jj[-1]))
    ax.add_patch(rect)
    ax.add_patch(patch)
    plt.colorbar()
    plt.title(r"$|f_{\bf{q}}|^2$")
    plt.xlabel("$q_x$")
    plt.ylabel("$q_y$")
    
    ax = plt.subplot(122)
    plt.imshow(np.fft.fftshift(g1(0)),extent=(ii[0],ii[-1],jj[0],jj[-1]))
    rect = patches.Rectangle((-KIMAX,-KJMAX),KJMAX*2+1,KIMAX*2+1,linewidth=linewidth,edgecolor='w',facecolor='none')
    patch= patches.Circle((0,0),35,linewidth=linewidth,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.add_patch(patch)
    plt.colorbar()
    plt.title(r"$a_{\bf{q}}$")
    plt.xlabel("$q_x$")
    plt.ylabel("$q_y$")
    
    plt.tight_layout()
    
    
    if SAVE_FIGS:
        fig.savefig("plots/form_factor.pdf")
    else:
        plt.show()

