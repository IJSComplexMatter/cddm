"""
Correlation data IO function
"""

import os
import numpy as np

RAW_NAMES = "corr", "count", "sq", "m1", "m2"
BG_NAMES = "bg1", "bg2"
VAR_NAMES = "var1", "var2"
CORR_NAMES = "t","corr"

class DataNotFoundError(Exception):
    pass

def _inspect_tuple_data(data, count = None):
    if not isinstance(data, tuple):
        raise ValueError("Input data must be a tuple.")
    elif count is not None:
        if len(data) != count:
            raise ValueError(f"Input data must be a tuple of length {count}.")
    for i, value in enumerate(data):
        if not isinstance(value, np.ndarray) and value is not None:
            raise ValueError(f"Input data element [{i}] is not a numpy array.")

def _raw_data_type_dir(directory):
    """Returns raw data type string ("multitau", "linear", or "invalid") of a
    given directory data. Directory must exist, otherwise it raises 
    DataNotFoundError exception.
    """
    if not os.path.exists(directory):
        raise DataNotFoundError(f"Data folder '{directory}' does not exists")
    if os.path.exists(os.path.join(directory, "raw_fast.npz")) and os.path.exists(os.path.join(directory, "raw_slow.npz")):
        return "multitau"
    elif os.path.exists(os.path.join(directory, "raw.npz")):
        return "linear"
    else:
        return "invalid"

def raw_data_type(data):
    if isinstance(data,str):
        return _raw_data_type_dir(data)
    else:
        try:
            if isinstance(data, tuple) and len(data) == 2:
                _inspect_tuple_data(data[0], 5)
                _inspect_tuple_data(data[1], 5)
                return "multitau"
            else:
                _inspect_tuple_data(data, 5)
                return "linear"
        except ValueError:
            return "invalid"            
    
def has_stats(directory):
    """Returns True if directory data structure holds stats data.
    Directory must exist, otherwise it raises DataNotFoundError exception.
    """
    if not os.path.exists(directory):
        raise DataNotFoundError(f"Data folder '{directory}' does not exists")    
    if os.path.exists(os.path.join(directory, "bg.npz")) and os.path.exists(os.path.join(directory, "var.npz")):
        return True
    else:
        return False
    
def has_raw_data(directory):
    """Returns True if directory data structure holds raw data.
    Directory must exist, otherwise it raises DataNotFoundError exception.
    """
    if raw_data_type(directory) != "invalid":
        return True
    else:
        return False
    
def has_corr_data(directory):
    """Returns True if directory data structure holds correlation data.
    Directory must exist, otherwise it raises DataNotFoundError exception.
    """
    if not os.path.exists(directory):
        raise DataNotFoundError(f"Data folder '{directory}' does not exists")    
    if os.path.exists(os.path.join(directory, "corr.npz")) :
        return True
    else:
        return False
    
def _load_npz_data(directory, base_name, key_names):
    fname = os.path.join(directory, base_name)
    out = np.load(fname)
    return tuple((out.get(key) for key in key_names))

def _save_npz_data(directory, base_name, key_names, data, count = None, compress = False):
    _inspect_tuple_data(data,count)
    
    fname = os.path.join(directory, base_name)
    data_dict = {key : value for key, value in zip(key_names, data) if value is not None}
    
    if compress == True:
        np.savez_compressed(fname, **data_dict)
    else:
        np.savez(fname, **data_dict)
        
def save_raw(directory, data):
    """Saves raw data tuple of length 5: (corr, count, sq, m1, m2) to a directory."""
    if not os.path.exists(directory):
        os.mkdir(directory)   
    if len(data) == 2:
        #multitau data
        data_fast, data_slow = data
        _save_npz_data(directory, "raw_fast.npz", RAW_NAMES, data, count = 5)
        _save_npz_data(directory,"raw_slow.npz", RAW_NAMES, data, count = 5)
    else:
        _save_npz_data(directory,"raw.npz", RAW_NAMES, data, count = 5)
        
def save_stats(directory, stats):
    """Saves stats data tuple of length 2: (bg,var) to a directory."""
    bg, var = stats
    if not isinstance(bg, tuple):
        #it must be a single camera data. We make it a single-element tuple
        bg = (bg,)
        count = 1
    else:
        count = 2
        
    _save_npz_data(directory,"bg.npz", BG_NAMES, bg, count = count)
    
    if not isinstance(var, tuple):
        #it must be a single camera data. We make it a single-element tuple
        var = (var,)
        count = 1
    else:
        count = 2

    _save_npz_data(directory, "var.npz",  VAR_NAMES, var, count = count)  

def save_corr(directory, data):
    """Saves stats data tuple of length 2: (t,data) to a directory."""
    _save_npz_data(directory,"corr.npz", CORR_NAMES, data, count = 2)
    
def load_raw(directory):
    """loads raw correlation data tuple of length 5: (corr, count, sq, m1, m2) 
    from a given directory. Raises DataNotFoundError if data does not exist,"""
    if raw_data_type(directory) == "multitau":
        return _load_npz_data(directory, "raw_fast.npz", RAW_NAMES), _load_npz_data(directory, "raw_slow.npz", RAW_NAMES)
    elif  raw_data_type(directory) == "linear":
        return _load_npz_data(directory, "raw.npz", RAW_NAMES)
    else:
        raise DataNotFoundError(f"No raw data found in '{directory}' data structure")

def load_stats(directory):
    """loads stats tuple of length 2: (bg, var) from a given directory. 
    Raises DataNotFoundError if data does not exist."""
    if has_stats(directory):
        bg = _load_npz_data(directory, "bg.npz", BG_NAMES)
        var = _load_npz_data(directory, "var.npz", VAR_NAMES)
        
        if bg[1] is None:
            #it must be a single camera data, so only return first element
            bg = bg[0]
        if var[1] is None:
            #it must be a single camera data, so only return first element
            var = var[0]
        return bg, var
    else:
        raise DataNotFoundError(f"No stats found in '{directory}' data structure")

def load_corr(directory):
    """loads correlation data tuple of length 2: (t, data) from a given directory. 
    Raises DataNotFoundError if data does not exist."""
    if has_corr_data(directory):
        return _load_npz_data(directory, "corr.npz", CORR_NAMES)  
    else:
        raise DataNotFoundError(f"No correlation data found in '{directory}' data structure")

