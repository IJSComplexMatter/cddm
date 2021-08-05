import unittest
import numpy as np
import cddm.core as core
from cddm.conf import FDTYPE, CDTYPE
from cddm.video import fromarrays

#test arrays
a = np.asarray([1.,2,3,4])
b = np.asarray([5,6,7,8])
a, b = a + 1j*b, b +1j*b
t1 = [1,3,7,8]
t2 = [2,4,6,8]
#results fo calculations
cross_a_b_complex = np.array([244.-104.j, 172. -84.j, 106. -60.j,  48. -32.j,  60. -20.j,
       122. -44.j, 184. -72.j],CDTYPE)
cross_a_b = cross_a_b_complex[0:4].real.copy()
cross_a_b[1:] += np.conj(cross_a_b_complex[-1:-4:-1]).real

cross_a_b_complex_3 = np.array([244.-104.j, 172. -84.j, 106. -60.j, 
       122. -44.j, 184. -72.j],CDTYPE)


cross_a_b_t1_t2_complex = np.array([ 96.-32.j, 158.-76.j,   0. +0.j,  92.-52.j,   0. +0.j, 106.-60.j,
         0. +0.j,  48.-32.j,   0. +0.j,  60.-20.j,  50.-20.j,  72.-24.j,
        60.-24.j,  84.-28.j, 110.-48.j],CDTYPE)
cross_a_b_t1_t2_complex_3 = np.array([ 96.-32.j, 158.-76.j,   0. +0.j,   
         84.-28.j, 110.-48.j],CDTYPE)
cross_a_b_t1_t2 = cross_a_b_t1_t2_complex[0:8].real.copy()
cross_a_b_t1_t2[1:] += np.conj(cross_a_b_t1_t2_complex[-1:-8:-1]).real

auto_a_complex = np.array([204. +0.j, 148.-12.j,  94.-16.j,  44.-12.j],CDTYPE)
auto_a = auto_a_complex.real

auto_a_t1_complex = np.array([204. +0.j,  68. -4.j,  32. -4.j,   0. +0.j,  48. -4.j,  56. -8.j,
        38. -8.j,  44.-12.j],CDTYPE)
auto_a_t1 = auto_a_t1_complex.real

auto_sum_a = np.array([10. +26.j ,  7.5+19.5j,  5. +13.j ,  2.5 +6.5j],CDTYPE)
auto_sum_a_t1 = np.array([10. +26.j ,  3.5 +7.5j,  1.5 +5.5j,  0.  +0.j ,  2.5 +6.5j,
        3.  +7.j ,  2.  +6.j ,  2.5 +6.5j],CDTYPE)


cross_sum_a_complex = np.array([10.+26.j,  6.+18.j,  3.+11.j,  1. +5.j,  4. +8.j,  7.+15.j,
        9.+21.j],CDTYPE)
cross_sum_a_complex_3 = np.array([10.+26.j,  6.+18.j,  3.+11.j,  7.+15.j,
        9.+21.j],CDTYPE)

cross_sum_a = cross_sum_a_complex[0:4].copy()
cross_sum_a[1:] += cross_sum_a_complex[-1:-4:-1]

cross_sum_a_t1_t2_complex = np.array([4. +8.j, 6.+18.j, 0. +0.j, 3.+11.j, 0. +0.j, 3.+11.j, 0. +0.j,
       1. +5.j, 0. +0.j, 4. +8.j, 3. +7.j, 4. +8.j, 3. +7.j, 4. +8.j,
       5.+13.j],CDTYPE)

cross_sum_a_t1_t2_complex_3 = np.array([4. +8.j, 6.+18.j, 0. +0.j, 4. +8.j,
       5.+13.j],CDTYPE)

cross_sum_a_t1_t2 = cross_sum_a_t1_t2_complex[0:8].copy()
cross_sum_a_t1_t2[1:] +=  cross_sum_a_t1_t2_complex[-1:-8:-1]

cross_count_10_complex = np.array([10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  6,  7,
        8,  9])
cross_count_10_complex_5 = np.array([10,  9,  8,  7,  6,  6,  7,
        8,  9])
cross_count_10 = np.array([10, 18, 16, 14, 12, 10,  8,  6,  4,  2],FDTYPE)

cross_count_t1_t2_complex = np.array([1, 3, 0, 2, 0, 2, 0, 1, 0, 1, 1, 1, 1, 1, 2])
cross_count_t1_t2_complex_5 = np.array([1, 3, 0, 2, 0,  1, 1, 1, 2])
cross_count_t1_t2 = np.array([1, 5, 1, 3, 1, 3, 1, 1])

auto_count_10 = np.array([10,  9,  8,  7,  6,  5,  4,  3,  2,  1],FDTYPE)
auto_count_t1 = np.array([4, 1, 1, 0, 1, 1, 1, 1],FDTYPE)

np.random.seed(0)

a2 = [a,a]
b2 = [b,b]

test_data1 = np.random.randn(32,19,8) + np.random.randn(32,19,8)*1j
test_data2 = np.random.randn(32,19,8) + np.random.randn(32,19,8)*1j
test_data1 = np.array(test_data1, CDTYPE)
test_data2 = np.array(test_data2, CDTYPE)

test_mask = np.ones((19,8),bool)
test_mask[0] = False
test_mask[:,0::3] = False

def allclose(a,b, rtol = 1e-5, atol = 1e-8):
    if FDTYPE == "float32":
        return np.allclose(a,b,rtol = rtol/100, atol = atol/1000)
    else:
        return np.allclose(a,b,rtol = rtol, atol = atol)


class TestCorrelateDifference(unittest.TestCase):
    
    def setUp(self):
        pass  
    
    def test_auto_correlate_complex_fft(self):
        out = core.auto_correlate_fft(a, complex = True)
        self.assertTrue(allclose(out,auto_a_complex))
        out = core.auto_correlate_fft(a,t1, complex = True)
        self.assertTrue(allclose(out,auto_a_t1_complex, atol = 1e-6)) 
        out = core.auto_correlate_fft(a,t1, aout = out,  complex = True)
        self.assertTrue(allclose(out,auto_a_t1_complex*2,atol = 1e-6)) 
        
    def test_auto_correlate_fft(self):
        out = core.auto_correlate_fft(a)
        self.assertTrue(allclose(out,auto_a))
        out = core.auto_correlate_fft(a,t1)
        self.assertTrue(allclose(out,auto_a_t1, atol = 1e-6)) 
        out = core.auto_correlate_fft(a,t1, aout = out)
        self.assertTrue(allclose(out,auto_a_t1*2,atol = 1e-6)) 

    def test_auto_correlate_complex_fft2(self):    
        out = core.auto_correlate_fft(a2,axis = -1, complex = True)
        self.assertTrue(allclose(out[0],auto_a_complex))
        out = core.auto_correlate_fft(a2,t1,axis = -1, complex = True)
        self.assertTrue(allclose(out[0],auto_a_t1_complex, atol = 1e-6)) 
        out = core.auto_correlate_fft(a2,t1, axis = -1, aout = out, complex = True)
        self.assertTrue(allclose(out[0],auto_a_t1_complex*2, atol = 1e-6))   
        
    def test_auto_correlate_fft2(self):    
        out = core.auto_correlate_fft(a2,axis = -1)
        self.assertTrue(allclose(out[0],auto_a))
        out = core.auto_correlate_fft(a2,t1,axis = -1)
        self.assertTrue(allclose(out[0],auto_a_t1, atol = 1e-6)) 
        out = core.auto_correlate_fft(a2,t1, axis = -1, aout = out)
        self.assertTrue(allclose(out[0],auto_a_t1*2, atol = 1e-6))         
 
    def test_auto_correlate_complex_fft_n(self):
        out = core.auto_correlate_fft(a, n = 3, complex = True)
        self.assertTrue(allclose(out,auto_a_complex[0:3])) 
        out = core.auto_correlate_fft(a,t1,n = 3, complex = True)
        self.assertTrue(allclose(out,auto_a_t1_complex[0:3])) 
        out = core.auto_correlate_fft(a,t1,n = 3, aout = out, complex = True)
        self.assertTrue(allclose(out,auto_a_t1_complex[0:3]*2))         
 
    def test_auto_correlate_fft_n(self):
        out = core.auto_correlate_fft(a, n = 3)
        self.assertTrue(allclose(out,auto_a[0:3])) 
        out = core.auto_correlate_fft(a,t1,n = 3)
        self.assertTrue(allclose(out,auto_a_t1[0:3])) 
        out = core.auto_correlate_fft(a,t1,n = 3, aout = out)
        self.assertTrue(allclose(out,auto_a_t1[0:3]*2)) 

    def test_auto_correlate_complex_fft_n2(self):
        out = core.auto_correlate_fft(a2, axis = -1, n = 3, complex = True)
        self.assertTrue(allclose(out[0],auto_a_complex[0:3])) 
        out = core.auto_correlate_fft(a2,t1,n = 3, axis = -1, complex = True)
        self.assertTrue(allclose(out[0],auto_a_t1_complex[0:3])) 
        out = core.auto_correlate_fft(a2,t1,n = 3, axis = -1, aout = out, complex = True)
        self.assertTrue(allclose(out[0],auto_a_t1_complex[0:3]*2)) 

    def test_auto_correlate_fft_n2(self):
        out = core.auto_correlate_fft(a2, axis = -1, n = 3)
        self.assertTrue(allclose(out[0],auto_a[0:3])) 
        out = core.auto_correlate_fft(a2,t1,n = 3, axis = -1)
        self.assertTrue(allclose(out[0],auto_a_t1[0:3])) 
        out = core.auto_correlate_fft(a2,t1,n = 3, axis = -1, aout = out)
        self.assertTrue(allclose(out[0],auto_a_t1[0:3]*2)) 

    def test_auto_correlate_complex(self):
        out = core.auto_correlate(a, complex = True)
        self.assertTrue(allclose(out,auto_a_complex)) 
        out = core.auto_correlate(a,t1, complex = True)
        self.assertTrue(allclose(out,auto_a_t1_complex)) 
        out = core.auto_correlate(a,t1, aout = out, complex = True)
        self.assertTrue(allclose(out,auto_a_t1_complex*2)) 

    def test_auto_correlate(self):
        out = core.auto_correlate(a)
        self.assertTrue(allclose(out,auto_a)) 
        out = core.auto_correlate(a,t1)
        self.assertTrue(allclose(out,auto_a_t1)) 
        out = core.auto_correlate(a,t1, aout = out)
        self.assertTrue(allclose(out,auto_a_t1*2)) 
 
    def test_auto_correlate_complex2(self):
        out = core.auto_correlate(a2, axis = -1, complex = True)
        self.assertTrue(allclose(out[0],auto_a_complex)) 
        out = core.auto_correlate(a2,t1, axis = -1, complex = True)
        self.assertTrue(allclose(out[0],auto_a_t1_complex)) 
        out = core.auto_correlate(a2,t1, axis = -1, aout = out, complex = True)
        self.assertTrue(allclose(out[0],auto_a_t1_complex*2)) 
               
 
    def test_auto_correlate2(self):
        out = core.auto_correlate(a2, axis = -1)
        self.assertTrue(allclose(out[0],auto_a)) 
        out = core.auto_correlate(a2,t1, axis = -1)
        self.assertTrue(allclose(out[0],auto_a_t1)) 
        out = core.auto_correlate(a2,t1, axis = -1, aout = out)
        self.assertTrue(allclose(out[0],auto_a_t1*2)) 

    def test_auto_correlate_complex_n(self):
        out = core.auto_correlate(a, n = 3, complex = True)
        self.assertTrue(allclose(out,auto_a_complex[0:3])) 
        out = core.auto_correlate(a,t1,n = 3, complex = True)
        self.assertTrue(allclose(out,auto_a_t1_complex[0:3])) 
        out = core.auto_correlate(a,t1,n = 3, aout = out, complex = True)
        self.assertTrue(allclose(out,auto_a_t1_complex[0:3]*2)) 
        
    def test_auto_correlate_n(self):
        out = core.auto_correlate(a, n = 3)
        self.assertTrue(allclose(out,auto_a[0:3])) 
        out = core.auto_correlate(a,t1,n = 3)
        self.assertTrue(allclose(out,auto_a_t1[0:3])) 
        out = core.auto_correlate(a,t1,n = 3, aout = out)
        self.assertTrue(allclose(out,auto_a_t1[0:3]*2)) 

    def test_auto_correlate_complex_n2(self):
        out = core.auto_correlate(a2, n = 3,axis = -1, complex = True)
        self.assertTrue(allclose(out[0],auto_a_complex[0:3])) 
        out = core.auto_correlate(a2,t1,n = 3, axis = -1, complex = True)
        self.assertTrue(allclose(out[0],auto_a_t1_complex[0:3])) 
        out = core.auto_correlate(a2,t1,n = 3, aout = out, axis = 1, complex = True)
        self.assertTrue(allclose(out[0],auto_a_t1_complex[0:3]*2)) 
        
    def test_auto_correlate_n2(self):
        out = core.auto_correlate(a2, n = 3,axis = -1)
        self.assertTrue(allclose(out[0],auto_a[0:3])) 
        out = core.auto_correlate(a2,t1,n = 3, axis = -1)
        self.assertTrue(allclose(out[0],auto_a_t1[0:3])) 
        out = core.auto_correlate(a2,t1,n = 3, aout = out, axis = 1)
        self.assertTrue(allclose(out[0],auto_a_t1[0:3]*2)) 

    def test_cross_correlate_complex_fft(self):
        out = core.cross_correlate_fft(a,b, complex = True)
        self.assertTrue(allclose(out,cross_a_b_complex)) 
        out = core.cross_correlate_fft(a,b,t1,t2, complex = True)
        self.assertTrue(allclose(out,cross_a_b_t1_t2_complex)) 
        out = core.cross_correlate_fft(a,b,t1,t2, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_a_b_t1_t2_complex*2)) 
        
    def test_cross_correlate_fft(self):
        out = core.cross_correlate_fft(a,b)
        self.assertTrue(allclose(out,cross_a_b)) 
        out = core.cross_correlate_fft(a,b,t1,t2)
        self.assertTrue(allclose(out,cross_a_b_t1_t2)) 
        out = core.cross_correlate_fft(a,b,t1,t2, aout = out)
        self.assertTrue(allclose(out,cross_a_b_t1_t2*2)) 


    def test_cross_correlate_complex_fft2(self):
        out = core.cross_correlate_fft(a2,b2,axis = 1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_complex)) 
        out = core.cross_correlate_fft(a2,b2,t1,t2,axis = 1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2_complex)) 
        out = core.cross_correlate_fft(a2,b2,t1,t2, aout = out,axis = -1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2_complex*2)) 
        
    def test_cross_correlate_fft2(self):
        out = core.cross_correlate_fft(a2,b2,axis = 1)
        self.assertTrue(allclose(out[0],cross_a_b)) 
        out = core.cross_correlate_fft(a2,b2,t1,t2,axis = 1)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2)) 
        out = core.cross_correlate_fft(a2,b2,t1,t2, aout = out,axis = -1)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2*2)) 

    def test_cross_correlate_complex_fft_n(self):
        out = core.cross_correlate_fft(a,b, n = 3, complex = True)
        self.assertTrue(allclose(out,cross_a_b_complex_3)) 
        out = core.cross_correlate_fft(a,b,t1,t2, n = 3, complex = True)
        self.assertTrue(allclose(out,cross_a_b_t1_t2_complex_3)) 
        out = core.cross_correlate_fft(a,b,t1,t2, n = 3, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_a_b_t1_t2_complex_3*2)) 
        
    def test_cross_correlate_fft_n(self):
        out = core.cross_correlate_fft(a,b, n = 3)
        self.assertTrue(allclose(out,cross_a_b[:3])) 
        out = core.cross_correlate_fft(a,b,t1,t2, n = 3)
        self.assertTrue(allclose(out,cross_a_b_t1_t2[:3])) 
        out = core.cross_correlate_fft(a,b,t1,t2, n = 3, aout = out)
        self.assertTrue(allclose(out,cross_a_b_t1_t2[:3]*2)) 

    def test_cross_correlate_complex_fft_n2(self):
        out = core.cross_correlate_fft(a2,b2, n = 3 ,axis = -1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_complex_3)) 
        out = core.cross_correlate_fft(a2,b2,t1,t2, n = 3, axis = -1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2_complex_3)) 
        out = core.cross_correlate_fft(a2,b2,t1,t2, n = 3, aout = out, axis = -1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2_complex_3*2)) 

    def test_cross_correlate_fft_n2(self):
        out = core.cross_correlate_fft(a2,b2, n = 3 ,axis = -1)
        self.assertTrue(allclose(out[0],cross_a_b[:3])) 
        out = core.cross_correlate_fft(a2,b2,t1,t2, n = 3, axis = -1)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2[:3])) 
        out = core.cross_correlate_fft(a2,b2,t1,t2, n = 3, aout = out, axis = -1)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2[:3]*2)) 

    def test_cross_correlate_complex(self):
        out = core.cross_correlate(a,b, complex = True)
        self.assertTrue(allclose(out,cross_a_b_complex)) 
        out = core.cross_correlate(a,b,t1,t2, complex = True)
        self.assertTrue(allclose(out,cross_a_b_t1_t2_complex)) 
        out = core.cross_correlate(a,b,t1,t2, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_a_b_t1_t2_complex*2)) 

    def test_cross_correlate(self):
        out = core.cross_correlate(a,b)
        self.assertTrue(allclose(out,cross_a_b)) 
        out = core.cross_correlate(a,b,t1,t2)
        self.assertTrue(allclose(out,cross_a_b_t1_t2)) 
        out = core.cross_correlate(a,b,t1,t2, aout = out)
        self.assertTrue(allclose(out,cross_a_b_t1_t2*2)) 

    def test_cross_correlate_complex2(self):
        out = core.cross_correlate(a2,b2,axis = -1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_complex)) 
        out = core.cross_correlate(a2,b2,t1,t2,axis = -1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2_complex)) 
        out = core.cross_correlate(a2,b2,t1,t2, aout = out,axis = -1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2_complex*2)) 

    def test_cross_correlate2(self):
        out = core.cross_correlate(a2,b2,axis = -1)
        self.assertTrue(allclose(out[0],cross_a_b)) 
        out = core.cross_correlate(a2,b2,t1,t2,axis = -1)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2)) 
        out = core.cross_correlate(a2,b2,t1,t2, aout = out,axis = -1)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2*2)) 

    def test_cross_correlate_complex_n(self):
        out = core.cross_correlate(a,b, n = 3, complex = True)
        self.assertTrue(allclose(out,cross_a_b_complex_3)) 
        out = core.cross_correlate(a,b,t1,t2, n = 3, complex = True)
        self.assertTrue(allclose(out,cross_a_b_t1_t2_complex_3)) 
        out = core.cross_correlate(a,b,t1,t2, n = 3, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_a_b_t1_t2_complex_3*2)) 
        
    def test_cross_correlate_n(self):
        out = core.cross_correlate(a,b, n = 3)
        self.assertTrue(allclose(out,cross_a_b[:3])) 
        out = core.cross_correlate(a,b,t1,t2, n = 3)
        self.assertTrue(allclose(out,cross_a_b_t1_t2[:3])) 
        out = core.cross_correlate(a,b,t1,t2, n = 3, aout = out)
        self.assertTrue(allclose(out,cross_a_b_t1_t2[:3]*2)) 

    def test_cross_correlate_complex_n2(self):
        out = core.cross_correlate(a2,b2, n = 3,axis = -1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_complex_3)) 
        out = core.cross_correlate(a2,b2,t1,t2, n = 3, axis = -1, complex = True)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2_complex_3)) 
        out = core.cross_correlate(a2,b2,t1,t2, n = 3, aout = out, axis = -1, complex = True)
        self.assertTrue(allclose(out,cross_a_b_t1_t2_complex_3*2)) 
        
    def test_cross_correlate_n2(self):
        out = core.cross_correlate(a2,b2, n = 3,axis = -1)
        self.assertTrue(allclose(out[0],cross_a_b[:3])) 
        out = core.cross_correlate(a2,b2,t1,t2, n = 3, axis = -1)
        self.assertTrue(allclose(out[0],cross_a_b_t1_t2[:3])) 
        out = core.cross_correlate(a2,b2,t1,t2, n = 3, aout = out, axis = -1)
        self.assertTrue(allclose(out,cross_a_b_t1_t2[:3]*2)) 

class TestSum(unittest.TestCase):
    def test_auto_sum(self):
        out = core.auto_sum(a)
        self.assertTrue(allclose(out,auto_sum_a))
        out = core.auto_sum(a,t1)
        self.assertTrue(allclose(out,auto_sum_a_t1)) 
        out = core.auto_sum(a,t1, aout = out)
        self.assertTrue(allclose(out,auto_sum_a_t1*2)) 

    def test_auto_sum_n(self):
        out = core.auto_sum(a, n = 3)
        self.assertTrue(allclose(out,auto_sum_a[0:3]))
        out = core.auto_sum(a,t1, n = 3)
        self.assertTrue(allclose(out,auto_sum_a_t1[0:3])) 
        out = core.auto_sum(a,t1,  aout = out)
        self.assertTrue(allclose(out,auto_sum_a_t1[0:3]*2)) 
        out = core.auto_sum(a,t1, n = 3, aout = out)
        self.assertTrue(allclose(out,auto_sum_a_t1[0:3]*3)) 
        
    def test_auto_sum_fft(self):
        out = core.auto_sum_fft(a,t1)
        self.assertTrue(allclose(out,auto_sum_a_t1)) 
        out = core.auto_sum_fft(a,t1, aout = out)
        self.assertTrue(allclose(out,auto_sum_a_t1*2)) 
        
    def test_auto_sum_fft_n(self):
        out = core.auto_sum_fft(a,t1, n = 3)
        self.assertTrue(allclose(out,auto_sum_a_t1[0:3])) 
        out = core.auto_sum_fft(a,t1, n =3, aout = out)
        self.assertTrue(allclose(out,auto_sum_a_t1[0:3]*2)) 
        out = core.auto_sum_fft(a,t1, aout = out)
        self.assertTrue(allclose(out,auto_sum_a_t1[0:3]*3))

    def test_cross_sum_complex(self):
        out = core.cross_sum(a, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_complex))
        out = core.cross_sum(a,t1,t2, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex)) 
        out = core.cross_sum(a,t1,t2, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex*2)) 

    def test_cross_sum(self):
        out = core.cross_sum(a)
        self.assertTrue(allclose(out,cross_sum_a))
        out = core.cross_sum(a,t1,t2)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2)) 
        out = core.cross_sum(a,t1,t2, aout = out)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2*2)) 

    def test_cross_sum_complex_n(self):
        out = core.cross_sum(a, n=3, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_complex_3))
        out = core.cross_sum(a,t1,t2, n = 3, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex_3)) 
        out = core.cross_sum(a,t1,t2, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex_3*2)) 

    def test_cross_sum_n(self):
        out = core.cross_sum(a, n=3)
        self.assertTrue(allclose(out,cross_sum_a[0:3]))
        out = core.cross_sum(a,t1,t2, n = 3)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2[0:3])) 
        out = core.cross_sum(a,t1,t2, aout = out)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2[0:3]*2)) 

    def test_cross_sum_fft_complex(self):
        out = core.cross_sum_fft(a,t1,t2, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex)) 
        out = core.cross_sum_fft(a,t1,t2, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex*2)) 

        out = core.cross_sum_fft(a.real,t1,t2, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex.real)) 
        out = core.cross_sum_fft(a.real,t1,t2, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex.real*2)) 

    def test_cross_sum_fft(self):
        out = core.cross_sum_fft(a,t1,t2)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2)) 
        out = core.cross_sum_fft(a,t1,t2, aout = out)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2*2)) 
        
        out = core.cross_sum_fft(a.real,t1,t2)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2.real)) 
        out = core.cross_sum_fft(a.real,t1,t2, aout = out)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2.real*2)) 

    def test_cross_sum_fft_complex_n(self):
        out = core.cross_sum_fft(a,t1,t2, n = 3, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex_3)) 
        out = core.cross_sum_fft(a,t1,t2, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex_3*2)) 
        out = core.cross_sum_fft(a,t1,t2, n =3, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex_3*3)) 
        
        out = core.cross_sum_fft(a.real,t1,t2, n = 3, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex_3.real)) 
        out = core.cross_sum_fft(a.real,t1,t2, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex_3.real*2)) 
        out = core.cross_sum_fft(a.real,t1,t2, n =3, aout = out, complex = True)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2_complex_3.real*3)) 

    def test_cross_sum_fft_n(self):
        out = core.cross_sum_fft(a,t1,t2, n = 3)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2[0:3])) 
        out = core.cross_sum_fft(a,t1,t2, aout = out)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2[0:3]*2)) 
        out = core.cross_sum_fft(a,t1,t2, n =3, aout = out)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2[0:3]*3)) 
        
        out = core.cross_sum_fft(a.real,t1,t2, n = 3)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2.real[0:3])) 
        out = core.cross_sum_fft(a.real,t1,t2, aout = out)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2.real[0:3]*2)) 
        out = core.cross_sum_fft(a.real,t1,t2, n =3, aout = out)
        self.assertTrue(allclose(out,cross_sum_a_t1_t2.real[0:3]*3)) 
        
    def test_cross_sum_equivalence_ND(self):
        for axis in (0,1,2):
            t1  = np.arange(test_data1.shape[axis])
            t2  = np.arange(test_data1.shape[axis]) + 3
            out1 = core.cross_sum(test_data1,t1,t2, axis = axis)
            out2 = core.cross_sum_fft(test_data1,t1,t2, axis = axis)
            self.assertTrue(allclose(out1,out2)) 

class TestCount(unittest.TestCase):
    def test_cross_count_complex(self):
        out = core.cross_count(10, complex = True)
        self.assertTrue(allclose(out,cross_count_10_complex))
        out = core.cross_count(t1,t2, complex = True)
        self.assertTrue(allclose(out,cross_count_t1_t2_complex)) 
        out = core.cross_count(t1,t2, complex = True, aout = out)
        self.assertTrue(allclose(out,cross_count_t1_t2_complex*2))  
    
    def test_cross_count(self):
        out = core.cross_count(10)
        self.assertTrue(allclose(out,cross_count_10))
        out = core.cross_count(t1,t2)
        self.assertTrue(allclose(out,cross_count_t1_t2)) 
        out = core.cross_count(t1,t2, aout = out)
        self.assertTrue(allclose(out,cross_count_t1_t2*2))  

    def test_cross_count_complex_n(self):
        out = core.cross_count(10, n = 5, complex = True)
        self.assertTrue(allclose(out,cross_count_10_complex_5))
        out = core.cross_count(t1,t2,n=5, complex = True)
        self.assertTrue(allclose(out,cross_count_t1_t2_complex_5)) 
        out = core.cross_count(t1,t2, aout = out, complex = True)
        self.assertTrue(allclose(out,2*cross_count_t1_t2_complex_5))  

    def test_cross_count_n(self):
        out = core.cross_count(10, n = 5)
        self.assertTrue(allclose(out,cross_count_10[0:5]))
        out = core.cross_count(t1,t2,n=5)
        self.assertTrue(allclose(out,cross_count_t1_t2[0:5])) 
        out = core.cross_count(t1,t2, aout = out)
        self.assertTrue(allclose(out,2*cross_count_t1_t2[0:5]))  

    def test_auto_count(self):
        out = core.auto_count(10)
        self.assertTrue(allclose(out,auto_count_10))
        out = core.auto_count(t1)
        self.assertTrue(allclose(out,auto_count_t1)) 
        out = core.auto_count(t1, aout = out)
        self.assertTrue(allclose(out,auto_count_t1*2))  

    def test_auto_count_n(self):
        out = core.auto_count(10, n = 5)
        self.assertTrue(allclose(out,auto_count_10[0:5]))
        out = core.auto_count(t1, n = 5)
        self.assertTrue(allclose(out,auto_count_t1[:5])) 
        out = core.auto_count(t1, aout = out)
        self.assertTrue(allclose(out,2*auto_count_t1[:5]))   


class TestIcorr(unittest.TestCase):
    def test_cross_equivalence(self):
        for method in ("corr","fft"):
            bg,var = core.stats(test_data1, test_data2, axis = 0)
            data = core.ccorr(test_data1, test_data2,n = 8, norm = 1, method = method)
            out1 = core.normalize(data, bg, var)
            vid = fromarrays((test_data1, test_data2))
            data,bg,var = core.iccorr(vid, count = len(test_data1),chunk_size = 16,n = 8, norm = 1, method = method)
            out2 = core.normalize(data, bg, var)  
            self.assertTrue(allclose(out1, out2))              

    def test_auto_equivalence_3(self):
        for method in ("corr",):
            bg,var = core.stats(test_data1, axis = 0)
            data1 = core.ccorr(test_data1,test_data1, n = 8, norm = 3, method = method)
            out1 = core.normalize(data1, bg, var, norm = 3)
            data2,bg,var = core.iacorr(test_data1, n = 8, norm = 3, method = method)
            out2 = core.normalize(data2, bg, var, norm = 3)  
            self.assertTrue(allclose(out1, out2))    

    def test_auto_equivalence_2(self):
        for method in ("corr","fft","diff"):
            bg,var = core.stats(test_data1, axis = 0)
            data1 = core.acorr(test_data1, n = 8, norm = 2, method = method)
            out1 = core.normalize(data1, bg, var, norm = 2)
            data2,bg,var = core.iacorr(test_data1, n = 8, norm = 2, method = method)
            out2 = core.normalize(data2, bg, var, norm = 2)  
            self.assertTrue(allclose(out1, out2))    

def wrap(data, conj = True):
    n = (data.shape[-1]+1)//2

    out = data[...,0:n].copy()
    if conj == True:
        out[...,1:] += np.conj(data[...,-1:-n:-1])
    else:
        out[...,1:] += data[...,-1:-n:-1]

    return out


class TestCorr(unittest.TestCase):
    
    def setUp(self):
        pass      
    
    def test_ccorr_complex_irregular(self):
        t1 = np.arange(16)
        t2 = np.arange(16)
        d1 = core.ccorr(test_data1[0:16], test_data2[0:16], t1, t2, complex = True)
        d2 = core.ccorr(test_data1[0:16], test_data2[0:16], t1, t2)
        self.assertTrue(allclose(wrap(d1[0]).real,d2[0]))
        self.assertTrue(allclose(wrap(d1[1]).real,d2[1]))
        self.assertTrue(allclose(wrap(d1[2]).real,d2[2]))
        self.assertTrue(allclose(wrap(d1[3], conj = False),d2[3]))
        self.assertTrue(allclose(wrap(d1[4], conj = False),d2[4]))
        
    def test_ccorr_complex_regular(self):
        d1 = core.ccorr(test_data1[0:16], test_data2[0:16], complex = True, method = "fft")
        d2 = core.ccorr(test_data1[0:16], test_data2[0:16])
        self.assertTrue(allclose(wrap(d1[0]).real,d2[0]))
        self.assertTrue(allclose(wrap(d1[1]).real,d2[1]))
        self.assertTrue(allclose(wrap(d1[2]),d2[2]))        
        self.assertTrue(allclose(wrap(d1[3], conj = False),d2[3]))
        self.assertTrue(allclose(wrap(d1[4], conj = False),d2[4]))

    def test_ccorr_regular_2(self):
        for scale in (True, False):
            for mode in ("corr", "diff"):
                for axis in (0,1,2):
                    bg,var = core.stats(test_data1, test_data2, axis = axis)
                    data = core.ccorr(test_data1, test_data2, norm = 2, method = "fft", axis = axis)
                    self.out = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale)
        
                    data = core.ccorr(test_data1, test_data2, norm = 2, method = "corr", axis = axis)
                    out_other = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale)
                    
                    self.assertTrue(allclose(self.out, out_other))
        
                    data = core.ccorr(test_data1, test_data2, norm = 2, method = "diff", axis = axis)
                    out_other = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale)
                    
                    self.assertTrue(allclose(self.out, out_other))

    def test_ccorr_regular_2_mask(self):
        for scale in (True, False):
            for mode in ("corr", "diff"):
                axis = 0
                bg,var = core.stats(test_data1, test_data2, axis = axis)
                data = core.ccorr(test_data1, test_data2, norm = 2, method = "fft", axis = axis)
                self.out = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale, mask = test_mask)
    
                data = core.ccorr(test_data1, test_data2, norm = 2, method = "corr", axis = axis)
                out_other = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale, mask = test_mask)
                
                self.assertTrue(allclose(self.out, out_other))
    
                data = core.ccorr(test_data1, test_data2, norm = 2, method = "diff", axis = axis)
                out_other = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale, mask = test_mask)
                
                self.assertTrue(allclose(self.out, out_other))
                
    def test_acorr_regular_2(self):
        for scale in (True, False):
            for mode in ("corr", "diff"):
                for axis in (0,1,2):
                    bg,var = core.stats(test_data1, axis = axis)
                    data = core.ccorr(test_data1, test_data1, norm = 2, method = "fft", axis = axis)
                    self.out = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale)
        
                    data = core.acorr(test_data1,norm = 2, method = "corr", axis = axis)
                    out_other = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale)
                    
                    self.assertTrue(allclose(self.out, out_other))
        
                    data = core.acorr(test_data1,norm = 2, method = "diff", axis = axis)
                    out_other = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale)
                    
                    self.assertTrue(allclose(self.out, out_other))
    
    def test_ccorr_regular_6(self):
        for scale in (True, False):
            for mode in ("corr", "diff"):
                for axis in (0,1,2):
                    bg,var = core.stats(test_data1, test_data2, axis = axis)
                    data = core.ccorr(test_data1, test_data2, norm = 6, method = "fft", axis = axis)
                    self.out = core.normalize(data, bg, var, norm = 6, mode = mode, scale = scale)
        
                    data = core.ccorr(test_data1, test_data2, norm = 6, method = "corr", axis = axis)
                    out_other = core.normalize(data, bg, var, norm = 6, mode = mode, scale = scale)
                    
                    self.assertTrue(allclose(self.out, out_other))
        
                    data = core.ccorr(test_data1, test_data2, norm = 6, method = "diff", axis = axis)
                    out_other = core.normalize(data, bg, var, norm = 6, mode = mode, scale = scale)
                    
                    self.assertTrue(allclose(self.out, out_other))

                    
    def test_ccorr_regular_6_mask(self):
        for scale in (True, False):
            for mode in ("corr", "diff"):
                axis = 0
                bg,var = core.stats(test_data1, test_data2, axis = axis)
                data = core.ccorr(test_data1, test_data2, norm = 6, method = "fft", axis = axis)
                self.out = core.normalize(data, bg, var, norm = 6, mode = mode, scale = scale,mask = test_mask)
    
                data = core.ccorr(test_data1, test_data2, norm = 6, method = "corr", axis = axis)
                out_other = core.normalize(data, bg, var, norm = 6, mode = mode, scale = scale,mask = test_mask)
                
                self.assertTrue(allclose(self.out, out_other))
    
                data = core.ccorr(test_data1, test_data2, norm = 6, method = "diff", axis = axis)
                out_other = core.normalize(data, bg, var, norm = 6, mode = mode, scale = scale,mask = test_mask)
                
                self.assertTrue(allclose(self.out, out_other))

    def test_ccorr_regular_1(self):
        for scale in (True, False):
            for mode in ("corr", "diff"):
                for axis in (0,1,2):
                    bg,var = core.stats(test_data1, test_data2, axis = axis)
                    data = core.ccorr(test_data1, test_data2, norm = 1, method = "fft", axis = axis)
                    self.out = core.normalize(data, bg, var, norm = 1, mode = mode, scale = scale)
        
                    data = core.ccorr(test_data1, test_data2, norm = 1, method = "corr", axis = axis)
                    out_other = core.normalize(data, bg, var, norm = 1, mode = mode, scale = scale)
                    
                    self.assertTrue(allclose(self.out, out_other))

    def test_acorr_regular_1(self):
        for scale in (True, False):
            for mode in ("corr", "diff"):
                for axis in (0,1,2):
                    bg,var = core.stats(test_data1, axis = axis)
                    data = core.acorr(test_data1,  norm = 1, method = "fft", axis = axis)
                    self.out = core.normalize(data, bg, var, norm = 1, mode = mode, scale = scale)
        
                    data = core.acorr(test_data1,norm = 1, method = "corr", axis = axis)
                    out_other = core.normalize(data, bg, var, norm = 1, mode = mode, scale = scale)
                    
                    self.assertTrue(allclose(self.out, out_other))


                    
    def test_corr_regular_1_mask(self):
        for scale in (True, False):
            for mode in ("corr", "diff"):
                axis = 0
                bg,var = core.stats(test_data1, test_data2, axis = axis)
                data = core.ccorr(test_data1, test_data2, norm = 1, method = "fft", axis = axis)
                self.out = core.normalize(data, bg, var, norm = 1, mode = mode, scale = scale, mask = test_mask)
    
                data = core.ccorr(test_data1, test_data2, norm = 1, method = "corr", axis = axis)
                out_other = core.normalize(data, bg, var, norm = 1, mode = mode, scale = scale, mask = test_mask)
                
                self.assertTrue(allclose(self.out, out_other))                    

    def test_corr_regular_2(self):
        for scale in (True, False):
            for mode in ("corr", "diff"):
                for axis in (0,1,2):
                    bg,var = core.stats(test_data1, test_data2, axis = axis)
                    data = core.ccorr(test_data1, test_data2, norm = 2, method = "fft", axis = axis)
                    self.out = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale)
        
                    data = core.ccorr(test_data1, test_data2, norm = 2, method = "corr", axis = axis)
                    out_other = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale)
                    
                    self.assertTrue(allclose(self.out, out_other))
                    
    def test_corr_regular_2_mask(self):
        for scale in (True, False):
            for mode in ("corr", "diff"):
                bg,var = core.stats(test_data1, test_data2)
                data = core.ccorr(test_data1, test_data2, norm = 2, method = "fft")
                self.out = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale, mask = test_mask)
    
                data = core.ccorr(test_data1, test_data2, norm = 2, method = "corr")
                out_other = core.normalize(data, bg, var, norm = 2, mode = mode, scale = scale,mask = test_mask)
                
                self.assertTrue(allclose(self.out, out_other))
                

class TestRest(unittest.TestCase):
    def test_abs2(self):
        self.assertTrue(allclose(core.abs2(test_data1), np.abs(test_data1)**2))


                
if __name__ == "__main__":
    unittest.main()