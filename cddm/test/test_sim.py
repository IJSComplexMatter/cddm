"""tests for Brownian simulator"""

import unittest

from cddm.sim import plot_random_walk, create_random_times1,create_random_times2,\
     simple_brownian_video     , seed

class TestSim(unittest.TestCase):
    """Tests function validity"""
    def setUp(self):
        seed(0)
        
    def test_plot(self):
        plot_random_walk()
        
    def test_video(self):
        video = simple_brownian_video(range(10))
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