"""tests for video processing functions"""

import unittest
import numpy as np

import cddm.conf as conf

class TestConf(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_functions(self):
        self.assertTrue(isinstance(conf.detect_number_of_cores(),int))
        conf.disable_mkl_threading()
        conf.enable_mkl_threading()
        conf.print_config()
    
    def test_set_rfft2lib(self):
        conf.set_rfft2lib("numpy")
        self.assertEqual("numpy",conf.set_rfft2lib("scipy"))
        if conf.SCIPY_INSTALLED:
            self.assertEqual("scipy",conf.set_rfft2lib("mkl_fft"))
        conf.set_rfft2lib("pyfftw")
        with self.assertRaises(ValueError):
            conf.set_rfft2lib(1)

    def test_set_fftlib(self):
        conf.set_fftlib("numpy")
        self.assertEqual("numpy",conf.set_fftlib("scipy"))
        if conf.SCIPY_INSTALLED:
            self.assertEqual("scipy",conf.set_fftlib("mkl_fft"))
        conf.set_fftlib("pyfftw")
        with self.assertRaises(ValueError):
            conf.set_fftlib(1)
            
    def test_set_verbose(self):
        conf.set_verbose(0)
        self.assertEqual(0,conf.set_verbose(1))
        self.assertEqual(1,conf.set_verbose(2))
        self.assertEqual(2,conf.set_verbose(0))
        with self.assertRaises(ValueError):
            conf.set_verbose("a")      
        
if __name__ == "__main__":
    unittest.main()