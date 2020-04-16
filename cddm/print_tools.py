"""Message printing functions. cddm uses cddm.conf.CDDMConfig.verbose 
variable to define verbosity level and behavior of printing functions.

verbosity levels 1 and 2 enable printing, 0 disables printing messages.
 """
from __future__ import absolute_import, print_function, division

import time
import cddm.conf

def print1(*args, **kwargs):
    """prints level 1 messages"""
    if cddm.conf.CDDMConfig.verbose >= 1:
        print(*args,**kwargs)
        
def print2(*args,**kwargs):
    """prints level 2 messages"""
    if cddm.conf.CDDMConfig.verbose >= 2:
        print(*args,**kwargs)
        
def print_progress (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '='):
    """
    Call in a loop to create terminal progress bar
    """
    if cddm.conf.CDDMConfig.verbose >= 1:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    #        sys.stdout.write(s)
    #        sys.stdout.flush()
        # Print New Line on Complete
        if iteration == total: 
            print()
        
def print_frame_rate(n_frames, t0, t1 = None, message = "... processed"):
    """Prints calculated frame rate"""
    if cddm.conf.CDDMConfig.verbose >= 2:
        if t1 is None:
            t1 = time.time()#take current time
        print (message + " {0} frames with an average frame rate {1:.2f}".format(n_frames, n_frames/(t1-t0)))
        
def disable_prints():
    """Disable message printing. Returns previous verbosity level"""
    #set verbosity to negative value
    value = cddm.conf.CDDMConfig.verbose
    cddm.conf.CDDMConfig.verbose = - abs(value)
    return value
    
def enable_prints(level = None):
    """Enable message printing. Returns previous verbosity level"""
    #set verbosity to positve value
    value = cddm.conf.CDDMConfig.verbose if level is None else int(level)
    cddm.conf.CDDMConfig.verbose =  abs(cddm.conf.CDDMConfig.verbose)
    return value
                                        