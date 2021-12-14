"""
Data buffering.
"""

from queue import Queue
from threading import Event

# buffer queues placehold
BUFFER_QUEUE = {}

BUFFER_DATA = {}

BUFFER_CHANGED_EVENT = {}
# buffer callback placehold
CALLBACK = {}
# names of buffers currently in active use. Each buffer object can be used only once.
BUFFERED_KEYS = set()

        
def buffer_key(key = None):
    """Generate a unique buffer key, or inspect if name is valid"""
    if key is None:
        i = len(BUFFER_QUEUE)
        return i
    else:
        if key in BUFFER_QUEUE.keys():
            raise ValueError("Buffer key {} already exists!".format(key))
        return key

def set_callback(callback, key = None):
    if key is None:
        key = tuple(BUFFER_QUEUE.keys())[-1]
    if key not in BUFFER_QUEUE.keys():
        raise ValueError("You must connect a callback to an existing buffer!")
        
    if callback is None:
        callback = CALLBACK.get(key, lambda x,y : x)
    if callable(callback):
        CALLBACK[key] = callback
    else:
        raise ValueError("Callback must be a callback object")

def remove_callback(key = None):
    if key is None:
        CALLBACK.clear()
    else:
        CALLBACK.pop(key,None)

def get_callback(key):
    return CALLBACK.get(key)
         
def create_buffer(key = None, maxsize = 0):
    key = buffer_key(key)
    queue = Queue(maxsize)
    BUFFER_QUEUE[key] = queue 
    BUFFER_CHANGED_EVENT[key] = Event()
    
def get_buffer_queue(key):
    return BUFFER_QUEUE.get(key)

def get_buffer_changed_event(key):
    return BUFFER_CHANGED_EVENT.get(key)

def destroy_buffer(key = None):
    if key is None:
        BUFFER_QUEUE.clear()
        BUFFER_DATA.clear()
        CALLBACK.clear()
        BUFFERED_KEYS.clear()
        BUFFER_CHANGED_EVENT.clear()
    else:
        BUFFER_QUEUE.pop(key,None)
        BUFFER_DATA.pop(key,None)
        CALLBACK.pop(key,None)
        BUFFERED_KEYS.discard(key)
        BUFFER_CHANGED_EVENT.pop(key,None)

def get_data(key):
    queue = get_buffer_queue(key)
    if queue and not queue.empty():
        BUFFER_DATA[key] = queue.get()
        queue.task_done()
    #buffer data may be empty, so we get it with a get method
    return BUFFER_DATA.get(key)
             
def iter_data(key):
    queue = get_buffer_queue(key)
    if queue:
        while not queue.empty():
            data = queue.get() 
            yield data
        
def next_data(key):
    queue = get_buffer_queue(key)
    if queue and not queue.empty():
        return queue.get()
    
def buffered(iterable, key = None, callback =  None, selected = None, maxsize = 0):
    if key is None:
        key = buffer_key()
    if key not in BUFFER_QUEUE.keys():
        create_buffer(key, maxsize)
    queue = get_buffer_queue(key)
    event = get_buffer_changed_event(key)
    if key in BUFFERED_KEYS:
        raise ValueError("Buffer already used!")
    else:
        BUFFERED_KEYS.add(key)
    set_callback(callback, key)

    return queued(iterable, queue, event = event, selected = selected)

# def process_buffer(keys = None, mode = "first"):
#     if keys is None:
#         keys = CALLBACK.keys()
#     elif not hasattr(keys, "__iter__"):
#         keys = (keys,)
#     if mode == "all": 
#         _process_buffer_all(keys)
#     elif mode == "first":
#         _process_buffer_first(keys)
#     elif mode == "last":
#         _process_buffer_last(keys)
#     else:
#         raise ValueError("Invalid buffer processing mode")
            
# def _process_buffer_all(keys):
#     for key in keys:
#         callback = get_callback(key)
#         queue = get_buffer(key)               
#         while not queue.empty():
#             data = queue.get()
#             callback(*data)    

# def _process_buffer_first(keys):
#     for key in keys:
#         callback = get_callback(key)
#         queue = get_buffer(key)  
#         #read first element and process              
#         if not queue.empty():
#             data = queue.get()
#             callback(*data) 
#         #empty buffer with no processing
#         while not queue.empty():
#             data = queue.get()  

# def _process_buffer_last(keys):
#     for key in keys:
#         callback = get_callback(key)
#         queue = get_buffer(key)   
#         data = None 
#         #read buffer data without processing
#         if not queue.empty():
#             data = queue.get()
#         #process only last obtained data
#         if data is not None:
#             callback(*data) 

def process_buffer(keys = None):
    if keys is None:
        keys = CALLBACK.keys()
    elif not hasattr(keys, "__iter__"):
        keys = (keys,)
    _process_buffer(keys)

 
def _process_buffer(keys):
    for key in keys:
        event = BUFFER_CHANGED_EVENT[key]
        if event.is_set():
            event.clear()
            callback = get_callback(key)
            data = get_data(key)  
            if data is not None:
                callback(*data) 
           
def queued(video, queue, event = None, selected = None, skip_if_full = True):
    if selected is None:
        def _is_selected(i):
            return True        
    elif callable(selected):
        def _is_selected(i):
            return selected(i)
    else:
        def _is_selected(i):
            return selected[i]      
    
    for i, frames in enumerate(video):
        if _is_selected(i):
            if queue.full() and skip_if_full:
                pass
            else:
                queue.put((i,frames))
            if event is not None:
                event.set()
        yield frames    



    
    