"""tests for map module"""

import unittest

from cddm.map import plot_indexmap, k_indexmap
class TestMap(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_plot(self):
        kmap = k_indexmap(33,32, angle = 0, sector = 5, kstep = 1.)
        plot_indexmap(kmap)

        
if __name__ == "__main__":
    unittest.main()