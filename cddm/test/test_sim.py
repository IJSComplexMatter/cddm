"""tests for Brownian simulator"""

import unittest

from cddm.sim import plot_random_walk, create_random_times1,create_random_times2,\
     simple_brownian_video   , seed, data_trigger, mirror, brownian_walk, brownian_particles

class TestSim(unittest.TestCase):
    """Tests function validity"""
    def setUp(self):
        seed(0)
        
    def test_mirror(self):
        self.assertAlmostEqual(mirror(2.,0.,2.), 0.)
        self.assertAlmostEqual(mirror(2.1,0.,2.), 0.1)
        self.assertAlmostEqual(mirror(-0.1,0.,1.), 0.9)
        self.assertAlmostEqual(mirror(0.,0.,1.), 0.)
        
    def test_brownian_walk(self):
        with self.assertRaises(ValueError):
            list(brownian_walk(((0,1),(2,3)),shape = (3,)))
        with self.assertRaises(ValueError):
            list(brownian_walk((0,1)))
            
    def test_brownian_particles(self):
        with self.assertRaises(ValueError):
            list(brownian_particles(num_particles = 2, x0 =((0,2),)))  
            
    def test_data_trigger(self):
        data = range(10)
        indices = [1,4,7]
        self.assertEqual(indices, [x for x in data_trigger(data, indices)])
        
    def test_plot(self):
        plot_random_walk()
        
    def test_video(self):
        video = simple_brownian_video(range(10), sigma = None)
        for frames in video:
            pass
    
    def test_dual_video(self):
        t1,t2 = create_random_times1(32,2)
        t1,t2 = create_random_times2(32,2)
        video = simple_brownian_video(t1,t2)
        for frames in video:
            pass
        
if __name__ == "__main__":
    unittest.main()