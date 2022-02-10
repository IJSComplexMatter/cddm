"""
Runtime functions. 
"""

from cddm.buffer import destroy_buffer, CALLBACK, process_buffer
import time
from queue import Queue, Full
from threading import Thread, Event
from cddm.viewer import pause

THREADS = {}
STOP_EVENTS = {}

def join_thread(name = None):
    """Joins a thread. If name is not specified,
    it joins all available threads."""
    if name is not None:
        try:
            THREADS[name].join(1)
            print(f"Thread {name} joined.")
        except RuntimeError:
            print(f"Error joining thread {name}")
            #in case we already started threads
            pass
        finally:
            THREADS.pop(name)
            STOP_EVENTS.pop(name)  
    else:
        names = tuple(THREADS.keys())
        for name in names:
            join_thread(name)    
        
def stop_thread(name = None):
    """Stops a running thread. If name is not specified,
    it stops all available threads."""
    if name is not None:
        STOP_EVENTS[name].set()
    else:
        names = tuple(STOP_EVENTS.keys())
        for name in names:
            stop_thread(name)
    
def start_thread(name = None):
    """Starts a thread with a specified name. If name is not specified,
    it starts all available threads."""
    if name is not None:
        try:
            THREADS[name].start()
            print(f"Thread {name} started")
        except RuntimeError:
            #in case we already started threads
            pass
    else:
        names = tuple(THREADS.keys())
        for name in names:
            start_thread(name)

def thread_name(name = None):
    """Returns a unique thread name. If name is specified, and is not unique,
    ValueError is raised."""
    if name is None:
        name = len(THREADS)
    if name in THREADS.keys():
        raise ValueError(f"Thread name {name} already exists.")
    return name

def threaded(iterable, queue_size = 0, block = True, name = None, queue = None, callback = None):
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
    queue: any
        An object that implements a queue interface. If not set, one is created from the
        queue_size parameter. If set, queue_size is ignored.
    
    Returns
    -------
    out : iterator
        A data iterator.
    """
    name = thread_name(name)
    if queue is None:
        q = Queue(queue_size)
    else:
        q = queue
    
    def worker(video, name, stop_event):
        try:
            for frames in video:
                try:
                    q.put(frames, block)
                except Full:
                    pass
                if stop_event.is_set():
                    break
        except BaseException as e:
            q.put(e)
        finally:
            print(f"Thread {name} stopped.")
            q.put(None)
    stop_event = Event()
    t = Thread(target=worker,args = (iterable,name,stop_event) ,daemon=True)
    THREADS[name] = t
    STOP_EVENTS[name] = stop_event
    
    def iterable(t,q):
        try:
            #make sure thread is started
            t.start()
        except RuntimeError:
            #if thread has already started just pass
            pass
        
        while True:
            out = q.get()
            q.task_done()
            if out is None:
                break
            elif isinstance(out, BaseException):
                raise out
            yield out
            
    return iterable(t,q)


def _run_buffered(iterable,  keys = None):
    try:     
        for out in iterable:
            process_buffer(keys = keys)
            yield out
    finally:
        destroy_buffer()

from pyface.api import GUI

def _run_buffered_threaded(iterable, keys = None, pause = None):
    q = Queue()
    name = thread_name()
    
    def worker(video, stop_event):
        try:
            for frames in video:
                if stop_event.is_set():
                    break
                q.put(frames)
        except BaseException as e:
            q.put(e)
        finally:
            q.put(None)
    stop_event = Event()
    t = Thread(target=worker,args = (iterable,stop_event) ,daemon=True)
    THREADS[name] = t
    STOP_EVENTS[name] = stop_event
    
    t.start()
    
    out = False #dummy  

    try:
        while True:
            out = False
            while not q.empty():
                out = q.get()

                if out is None:
                    break
                elif isinstance(out, BaseException):
                    raise out
                else:
                    yield out
                q.task_done()

            if out == False:
                time.sleep(1/30.)
            elif out is None:
                break
            else:
                process_buffer(keys = keys)
            
            if pause is not None:
                pause()

    finally:
        destroy_buffer()

def run_buffered(iterable, spawn = True, keys = None, pause = None):
    if spawn == True:
        return _run_buffered_threaded(iterable, keys =keys, pause = pause)
    else:
        return _run_buffered(iterable, keys =keys)

class RunningContext():
    def __init__(self, video, spawn = False, pause = None):
        self.video = video
        self.spawn = spawn

        self.iter = run_buffered(self.video, spawn = self.spawn, pause = pause)

    def __iter__(self):
        return self
    
    def __next__(self):
        return next(self.iter)
    
    def __enter__(self):
        return self.__iter__()

    def __exit__(self,exception_type, exception_value, exception_traceback):
        stop_thread()
        join_thread()
        destroy_buffer()

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
         
def running(video, spawn = False, pause = None):
    """Returns a running context 
    """
    if isinstance(video, RunningContext):
        raise ValueError("Cannot create a running context video.")
    return RunningContext(video, spawn, pause = pause)    

def run(video, spawn = True, callback = None, pause = None):
    """Runs the iterator and shows live graphs. 
    
    By default, a background thread is launched which performs the actual iteration
    and the main thread only reads data from queue. Any exception raised in the bacground
    thread is caught and re-raised in the main thread.
    """
    with running(video, spawn =  spawn, pause = pause) as video:
        for i, data in enumerate(video):
            if callback is not None:
                if not callback(i,data):
                    break
    return data

def process(iterable, spawn = True):
    """Consumes iterable and returns results queue"""
    queue = Queue(1)
    def worker(iterable, queue):
        try:
            for data in iterable:
                pass
        finally:
            queue.put(data)
    if spawn == True:
        t = Thread(target = worker, args = (iterable, queue), daemon = True)
        t.start()
    else:
        worker()
    return queue
        
if __name__ == "__main__":
    pass
    