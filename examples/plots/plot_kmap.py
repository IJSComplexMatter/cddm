"""Plots rfft2 k-values of the cropped and the original rfft2"""

#load simulation parameters
from examples.conf import SHAPE, KISIZE, KJSIZE, KIMAX, KJMAX

import matplotlib.pyplot as plt
import numpy as np
from cddm.map import rfft2_kangle

kmap, anglemap = rfft2_kangle(SHAPE[0], SHAPE[1]//2+1, shape = SHAPE)

kmap[:,KJMAX:] = np.nan
kmap[KIMAX+1:-KIMAX] = np.nan

plt.subplot(121)
im = plt.imshow(kmap)
plt.title("rfft2 - original")

kmap, anglemap = rfft2_kangle(KISIZE, KJSIZE, shape = SHAPE)

ax = plt.subplot(122)
im = plt.imshow(kmap)
plt.colorbar(im)
plt.title("rfft2 - cropped")

plt.show()