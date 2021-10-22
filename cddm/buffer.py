"""
Data buffering.
"""

from queue import Queue

BUFFER = {}
CALLBACK = {}
BUFFERED_KEYS = set()
        
def buffer_key(key = None):
    """Generate a unique buffer key, or inspect if name is valid"""
    if key is None:
        i = len(BUFFER)
        return i
    else:
        if key in BUFFER.keys():
            raise ValueError("Buffer key {} already exists!".format(key))
        return key

def set_callback(callback, key = None):
    if key is None:
        key = tuple(BUFFER.keys())[-1]
    if key not in BUFFER.keys():
        raise ValueError("You must connect a callback to an existing buffer!")
        
    if callback is None:
        callback = CALLBACK.get(key, lambda x : x)
    if callable(callback):
        CALLBACK[key] = callback
    else:
        raise ValueError("Callback must be a callback object")

def get_callback(key):
    return CALLBACK.get(key)
         
def set_buffer(key, maxsize = 0):
    key = buffer_key(key)
    queue = Queue(maxsize)
    BUFFER[key] = queue 

def get_buffer(key):
    return BUFFER.get(key)

def iter_data(key):
    queue = get_buffer(key)
    while not queue.empty():
        data = queue.get() 
        yield data
        
def next_data(key):
    queue = get_buffer(key)
    if not queue.empty():
        return queue.get()
    
def buffered(iterable, key = None, callback =  None, selected = None, maxsize = 0):
    if key is None:
        key = buffer_key()
    if key not in BUFFER.keys():
        set_buffer(key, maxsize)
    queue = get_buffer(key)
    if key in BUFFERED_KEYS:
        raise ValueError("Buffer already used!")
    else:
        BUFFERED_KEYS.add(key)
    set_callback(callback, key)
    return queued(iterable, queue, selected = selected)

def process_buffer(keys = None, mode = "first"):
    if keys is None:
        keys = CALLBACK.keys()
    elif not hasattr(keys, "__iter__"):
        keys = (keys,)
        
    if mode == "all": 
        _process_buffer_all(keys)
    elif mode == "first":
        _process_buffer_first(keys)
    elif mode == "last":
        _process_buffer_last(keys)
    else:
        raise ValueError("Invalid buffer processing mode")
            
def _process_buffer_all(keys):
    for key in keys:
        callback = get_callback(key)
        queue = get_buffer(key)               
        while not queue.empty():
            data = queue.get()
            callback(*data)    

def _process_buffer_first(keys):
    for key in keys:
        callback = get_callback(key)
        queue = get_buffer(key)  
        #read first element and process              
        if not queue.empty():
            data = queue.get()
            callback(*data) 
        #empty buffer with no processing
        while not queue.empty():
            data = queue.get()  

def _process_buffer_last(keys):
    for key in keys:
        callback = get_callback(key)
        queue = get_buffer(key)   
        data = None 
        #read buffer data without processing
        if not queue.empty():
            data = queue.get()
        #process only last obtained data
        if data is not None:
            callback(*data) 
            
def queued(video, queue, selected = None, skip_if_full = True):
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
        yield frames    

def clear_callback(key = None):
    if key is None:
        CALLBACK.clear()
    else:
        CALLBACK.pop(key)
                     
def clear_buffer(key = None):
    if key is None:
        BUFFER.clear()
        CALLBACK.clear()
        BUFFERED_KEYS.clear()
    else:
        BUFFER.pop(key)
        CALLBACK.pop(key)
        BUFFERED_KEYS.remove(key)
             

    
    