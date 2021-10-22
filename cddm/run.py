"""
Runtime functions. 
"""

from cddm.buffer import clear_buffer, clear_callback,CALLBACK, process_buffer
import time
from queue import Queue, Full
from threading import Thread, Event
from cddm.viewer import pause


THREADS = {}
STOP_EVENTS = {}

def join_threads():
    """Joins all threads"""
    try:
        for t in THREADS.values():
            t.join(1)
    finally:
        THREADS.clear()
        STOP_EVENTS.clear()
        
def stop_threads():
    """Stops all running threads"""
    print("Stopping threads")
    for stop_event in STOP_EVENTS.values():
        stop_event.set()
    
def start_threads(name = None):
    """Starts a thread with a specified name. If name is not specified,
    it starts all available threads."""
    if name is not None:
        THREADS[name].start()
        print(f"Thread {name} started")
    else:
        for name, t in THREADS.items():
            try:
                t.start()
                print(f"Thread {name} started")
            except RuntimeError:
                #in case we already started threads
                pass
    
def thread_name(name = None):
    """Returns a unique thread name. If name is specified, and is not unique,
    ValueError is raised."""
    if name is None:
        name = len(THREADS)
    if name in THREADS.keys():
        raise ValueError(f"Thread name {name} already exists.")
    return name

def threaded(iterable, queue_size = 0, block = True, name = None):
    """
    Returns a new iterator that runs the process in a background thread. 
    
    Parameters
    ----------
    video : iterator
        A multi-frame iterator
    queue_size : int
        Size of Queue object. If queue_size is <= 0, the queue size is infinite.
    block : bool
        Whether it blocks if queue is full or not. Data will be lost if block is set to 
        False and ques_size is finite and Full.
    name : any
        Name of the thread. It can be anyyhing that can be set as a dictionary key.
    
    Returns
    -------
    out : iterator
        A data iterator.
    """
    name = thread_name(name)
    q = Queue(queue_size)
    
    def worker(video, name, stop_event):
        try:
            for frames in video:
                try:
                    q.put(frames, block)
                except Full:
                    pass
                if stop_event.is_set():
                    break
        finally:
            print(f"Thread {name} stopped.")
            q.put(None)
    stop_event = Event()
    t = Thread(target=worker,args = (iterable,name,stop_event) ,daemon=True)
    THREADS[name] = t
    STOP_EVENTS[name] = stop_event
    
    def iterable(t,q):
        try:
            t.start()
        except RuntimeError:
            pass
        
        while True:
            out = q.get()
            q.task_done()
            if out is None:
                break
            yield out
            
    return iterable(t,q)

def _run_buffered(iterable, fps = None, keys = None, mode = "all", initialize = None, finalize = None):
    if keys is None:
        keys = CALLBACK.keys()
    
    t0 = None   
    i = 0
    try:     
        for out in iterable:
            if t0 is None:
                t0 = time.time()  
                
            if fps is None or time.time()-t0 >= i/fps: 
                if initialize is not None:
                    initialize()
                process_buffer(keys = keys, mode =  mode)
                if finalize is not None:
                    finalize()
                i+=1
            yield out
        
    finally:
        clear_buffer()

def _run_buffered_threaded(iterable, fps = None, keys = None, mode = "all", initialize = None, finalize = None):
    q = Queue()
    name = thread_name()
    
    def worker(video, stop_event):
        try:
            for frames in video:
                if stop_event.is_set():
                    break
                q.put(frames)
        finally:
            q.put(None)
    stop_event = Event()
    t = Thread(target=worker,args = (iterable,stop_event) ,daemon=True)
    THREADS[name] = t
    STOP_EVENTS[name] = stop_event
    
    t.start()
    
    out = False #dummy  
    t0 = None   
    i = 0
    try:
        while True:
            if fps is None:
                while not q.empty():
                    out = q.get()
                    if out is None:
                        break
                    else:
                        yield out
                    q.task_done()
                if initialize is not None:
                    initialize()
                process_buffer(keys = keys, mode =  mode)
                if finalize is not None:
                    finalize()
                i+=1
            else:
                if t0 is None:
                    t0 = time.time()
                while not q.empty():
                    out = q.get()
                    q.task_done()
    
                    if out is None:
                        break
                    else:
                        yield out
                dt = i/fps -  (time.time()-t0)
                if dt > 0:
                    time.sleep(dt)
                if initialize is not None:
                    initialize()
                process_buffer(keys = keys, mode =  mode)
                if finalize is not None:
                    finalize()
                i+=1
            if out is None:
                break
    finally:
        clear_buffer()

def run_buffered(iterable, spawn = True, fps = None, keys = None, mode = "all", initialize = None, finalize = None):
    if spawn == True:
        return _run_buffered_threaded(iterable, fps = fps, keys =keys, mode = mode, initialize = initialize, finalize = finalize)
    else:
        return _run_buffered(iterable, fps = fps, keys =keys, mode = mode, initialize = initialize, finalize = finalize)


class RunningContext():
    def __init__(self, video, fps = None, spawn = False):
        self.video = video
        self.spawn = spawn
        self.fps = fps
        self.iter = run_buffered(self.video, fps = self.fps, spawn = self.spawn, finalize = pause)

    def __iter__(self):
        return self
    
    def __next__(self):
        return next(self.iter)
    
    def __enter__(self):
        return self.__iter__()

    def __exit__(self,exception_type, exception_value, exception_traceback):
        stop_threads()
        join_threads()
        clear_callback()

        if exception_type in (KeyboardInterrupt,):
            #we do not want to raise exception if ctrl-c was pressed, so return True
            return False

def asrunning(video):
    """Converts iterable to a running iterable. 
    """
    if isinstance(video, RunningContext):
        return video
    else:
        return running(video)
        
def running(video, fps = None, spawn = False):
    """Returns a running context 
    """
    if isinstance(video, RunningContext):
        raise ValueError("Cannot create a running context video.")
    return RunningContext(video, fps, spawn)    

def run(video, fps = None, spawn = True):
    """Runs the iterator and shows live graphs. 
    
    By default, a background thread is launched which performs the iteration
    and the main thread is responsible for running live graphs. 
    """
    with running(video, fps = fps, spawn =  spawn) as video:
        for data in video:
            pass
    return data

if __name__ == "__main__":
    pass
    