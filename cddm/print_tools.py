import sys, time
import numba
import cddm.conf

_print = print

def print(*args, **kwargs):
    if cddm.conf.CDDMConfig.verbose >= 1:
        _print(*args,**kwargs)

            
def print_progress (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '='):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    if cddm.conf.CDDMConfig.verbose >= 1:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        _print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    #        sys.stdout.write(s)
    #        sys.stdout.flush()
        # Print New Line on Complete
        if iteration == total: 
            _print()
        
def print_frame_rate(n_frames, t0, t1 = None, message = "... processed"):
    if cddm.conf.CDDMConfig.verbose >= 2:
        if t1 is None:
            t1 = time.time()#take current time
        _print (message + " {0} frames with an average frame rate {1:.2f}".format(n_frames, n_frames/(t1-t0)))
        
