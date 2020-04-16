import unittest
import cddm.print_tools as pt
import cddm.conf as conf

class TestMulti(unittest.TestCase):
    
    def setUp(self):
       pass
   
    def test_print(self):
        pt.print1("test")
        pt.print2("test")
        pt.print_progress(4,10)
        
    def test_frame_rate(self):
        import time
        t0 = time.time()-1.
        pt.print_frame_rate(1024, t0)
        pt.print_frame_rate(1024, t0, time.time())
        
    def test_disable_prints(self):
        tmp = conf.CDDMConfig["verbose"]
        out = pt.disable_prints()
        self.assertEqual(out, tmp)
        self.assertEqual(0, conf.CDDMConfig["verbose"])
        pt.enable_prints(out)
        self.assertEqual(tmp, conf.CDDMConfig["verbose"])
        
if __name__ == "__main__":
    unittest.main()
        