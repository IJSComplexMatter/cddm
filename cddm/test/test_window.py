"""tests for window functions"""

import unittest
import numpy as np
from cddm.conf import FDTYPE
from cddm.window import blackman, gaussian, tukey, hann, plot_windows

def _r(shape, scale = 1.):
    """Returns radius array of a given shape."""
    ny,nx = [l/2 for l in shape]
    ay, ax = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    xx, yy = np.meshgrid(ax, ay, indexing = "xy")
    r = ((xx/(nx*scale))**2 + (yy/(ny*scale))**2) ** 0.5    
    return r

def _tukey(r,alpha = 0.1, rmax = 1.):
    out = np.ones(r.shape, FDTYPE)
    r = np.asarray(r, FDTYPE)
    alpha = alpha * rmax
    mask = r > rmax -alpha
    if alpha > 0.:
        tmp = 1/2*(1-np.cos(np.pi*(r-rmax)/alpha))
        out[mask] = tmp[mask]
    mask = (r>= rmax)
    out[mask] = 0.
    return out  

window_shapes = ((4,4), (16,11))        

class TestWindows(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_blackman(self):
        for shape in window_shapes:
            r = _r(shape)
            result = 0.42 + 0.5*np.cos(1*np.pi*r)+0.08*np.cos(2*np.pi*r)
            mask = (r>= 1.)
            result[mask] = 0.
            out = blackman(shape)
            self.assertTrue(np.allclose(result, out))
            out[...] = 0
            blackman(shape, out = out)
            self.assertTrue(np.allclose(result, out))

    def test_gaussian(self):
        for shape in window_shapes:
            sigma = 1.5
            r = _r(shape, sigma* (2**0.5))
            result = np.exp(-r**2)
            out = gaussian(shape, sigma)
            self.assertTrue(np.allclose(result, out))
            out[...] = 0
            gaussian(shape, sigma, out = out)
            self.assertTrue(np.allclose(result, out))
            
            with self.assertRaises(ValueError):
                gaussian(shape, -3.)
            
    def test_tukey(self):
        for shape in window_shapes:
            alpha = 0.8
            r = _r(shape)
            result = _tukey(r,alpha) 
            out = tukey(shape, alpha)
            self.assertTrue(np.allclose(result, out))
            out[...] = 0
            out = tukey(shape, alpha, out = out)
            self.assertTrue(np.allclose(result, out)) 
            
            with self.assertRaises(ValueError):
                tukey(shape, 1.01)
                
            with self.assertRaises(ValueError):
                tukey(shape, -0.2)           

    def test_hann(self):
        for shape in window_shapes:
            r = _r(shape)
            result = _tukey(r,1.) 
            out = hann(shape)
            self.assertTrue(np.allclose(result, out))
            out[...] = 0
            out = hann(shape, out = out)
            self.assertTrue(np.allclose(result, out))        
            
    def test_plot_window(self):
        self.assertEqual(None, plot_windows())
            
       
if __name__ == "__main__":
    unittest.main()