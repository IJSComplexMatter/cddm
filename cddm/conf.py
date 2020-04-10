"""
Configuration and constants
"""

from __future__ import absolute_import, print_function, division

import numpy as np
from functools import wraps
import os, warnings, shutil

#These will be defined later at runtime... here we hold reference to disable warnings in autoapi generation
I = None
CDTYPE = None 
IDTYPE = None
C = None
NUMBA_PARALLEL = None

try:
    from configparser import ConfigParser
except:
    #python 2.7
    from ConfigParser import ConfigParser

#: path to data files
DATAPATH = os.path.dirname(__file__)

def _read_environ_variable(name, default):
    """Reads environment variable"""
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)
        warnings.warn("Environment variable {0:s} was found, but its value is not valid!".format(name))

def get_home_dir():
    """
    Return user home directory
    """
    try:
        path = os.path.expanduser('~')
    except:
        path = ''
    for env_var in ('HOME', 'USERPROFILE', 'TMP'):
        if os.path.isdir(path):
            break
        path = os.environ.get(env_var, '')
    if path:
        return path
    else:
        raise RuntimeError('Please define environment variable $HOME')

HOMEDIR = get_home_dir() 
"""Users home directory"""

CDDM_CONFIG_DIR = os.path.join(HOMEDIR, ".cddm")
"""Config directory where cddm.ini lives"""

NUMBA_CACHE_DIR = os.path.join(CDDM_CONFIG_DIR, "numba_cache")
"""Numba cache directory, where compiled functions are written"""

if not os.path.exists(CDDM_CONFIG_DIR):
    try:
        os.makedirs(CDDM_CONFIG_DIR)
    except:
        warnings.warn("Could not create folder in user's home directory! Is it writeable?")
        NUMBA_CACHE_DIR = ""
        
CONF = os.path.join(CDDM_CONFIG_DIR, "cddm.ini")
"""Configuration file filename"""

CONF_TEMPLATE = os.path.join(DATAPATH, "cddm.ini")

config = ConfigParser()

if not os.path.exists(CONF):
    try:
        shutil.copy(CONF_TEMPLATE, CONF)
    except:
        warnings.warn("Could not copy config file in user's home directory! Is it writeable?")
        CONF = CONF_TEMPLATE
config.read(CONF)
    
def _readconfig(func, section, name, default):
    try:
        return func(section, name)
    except:
        return default


if NUMBA_CACHE_DIR != "":
    os.environ["NUMBA_CACHE_DIR"] = NUMBA_CACHE_DIR #set it to os.environ.. so that numba can use it

if _read_environ_variable("CDDM_TARGET_PARALLEL",
            default = _readconfig(config.getboolean, "numba", "parallel", False)):
    NUMBA_TARGET = "parallel"
    NUMBA_PARALLEL = True
else:
    NUMBA_TARGET = "cpu"
    NUMBA_PARALLEL = False

NUMBA_CACHE = False 
"""Specifiess whether we use numba caching or not"""  

_numba_0_39_or_greater = False
_numba_0_45_or_greater = False

try: 
    import numba as nb
    major, minor = nb.__version__.split(".")[0:2]
    if int(major) == 0 and int(minor) >=45:
        _numba_0_45_or_greater = True
    if int(major) == 0 and int(minor) >=39:
        _numba_0_39_or_greater = True
except:
    print("Could not determine numba version you are using, assuming < 0.39")


if _read_environ_variable("CDDM_NUMBA_CACHE",
            default = _readconfig(config.getboolean, "numba", "cache", True)):
    if NUMBA_PARALLEL == False:
        NUMBA_CACHE = True  
    else:
        #caching was not available for target parallel before 0.45 version
        if _numba_0_45_or_greater:
            NUMBA_CACHE = True

if _read_environ_variable("CDDM_FASTMATH",
        default = _readconfig(config.getboolean, "numba", "fastmath", False)):        
    NUMBA_FASTMATH = True
else:
    NUMBA_FASTMATH = False

def is_module_installed(name):
    """Checks whether module with name 'name' is istalled or not"""
    try:
        __import__(name)
        return True
    except ImportError:
        return False    
        
NUMBA_INSTALLED = is_module_installed("numba")
MKL_FFT_INSTALLED = is_module_installed("mkl_fft")
SCIPY_INSTALLED = is_module_installed("scipy")
CV2_INSTALLED = is_module_installed("cv2")

def detect_number_of_cores():
    """
    detect_number_of_cores()

    Detect the number of cores in this system.

    Returns
    -------
    out : int
        The number of cores in this system.

    """
    import os
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default

F32DTYPE = np.dtype("float32")
F64DTYPE = np.dtype("float64")

C64DTYPE = np.dtype("complex64")
C128DTYPE = np.dtype("complex128")

U8DTYPE = np.dtype("uint8")
U16DTYPE = np.dtype("uint16")
U32DTYPE = np.dtype("uint32")
U64DTYPE = np.dtype("uint64")

I32DTYPE = np.dtype("int32")
I64DTYPE = np.dtype("int64")


if _read_environ_variable("CDDM_DOUBLE_PRECISION",
  default =  _readconfig(config.getboolean, "core", "double_precision", True)):
    FDTYPE = F64DTYPE
    CDTYPE = C128DTYPE
    UDTYPE = U64DTYPE
    IDTYPE = I64DTYPE
    PRECISION = "double"
else:
    FDTYPE = F32DTYPE
    CDTYPE = C64DTYPE
    UDTYPE = U32DTYPE
    IDTYPE = I32DTYPE
    PRECISION = "single"
    
def disable_mkl_threading():
    """Disables mkl threading."""
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        warnings.warn("Cannot disable mkl threading, because 'mkl' module is not installed!")

def enable_mkl_threading():
    """Enables mkl threading."""
    try:
        import mkl
        mkl.set_num_threads(detect_number_of_cores())
    except ImportError:
        warnings.warn("Cannot enable mkl threading, because 'mkl' module is not installed!")

class CDDMConfig(object):
    """DDMM settings are here. You should use the set_* functions in the
    conf.py module to set these values"""
    def __init__(self):
        self.rfft2lib = "numpy"
        self.fftlib = "numpy"            
        self.verbose = 0
        self.cv2 = False
        
    def __getitem__(self, item):
        return self.__dict__[item]
        
    def __repr__(self):
        return repr(self.__dict__)

#: a singleton holding user configuration    
CDDMConfig = CDDMConfig()
 

def print_config():
    """Prints all compile-time and run-time configurtion parameters and settings."""
    options = {"PRECISION" : PRECISION, 
               "NUMBA_FASTMATH" :NUMBA_FASTMATH, "NUMBA_PARALLEL" : NUMBA_PARALLEL,
           "NUMBA_CACHE" : NUMBA_CACHE, "NUMBA_FASTMATH" : NUMBA_FASTMATH,
           "NUMBA_TARGET" : NUMBA_TARGET}
    options.update(CDDMConfig.__dict__)
    print(options)

def set_verbose(level):
    """Sets verbose level (0-2) used by compute functions."""
    out = CDDMConfig.verbose
    CDDMConfig.verbose = max(0,int(level))
    return out

def set_cv2(ok):
    """Enable/Disable cv2 rendering."""
    out = CDDMConfig.cv2
    if ok == True and not CV2_INSTALLED:
        warnings.warn("cv2 is not installed so it can not be used! Please install cv2.")            
    else:
        CDDMConfig.cv2 = bool(ok)
    return out
    
def set_nthreads(num):
    """Sets number of threads used by fft functions."""
    out = CDDMConfig.nthreads
    CDDMConfig.nthreads = max(1,int(num))
    return out
   
def set_fftlib(name = "numpy.fft"):
    """Sets fft library. Returns previous setting."""
    out, name = CDDMConfig.fftlib, str(name) 
    if name == "mkl_fft":
        if MKL_FFT_INSTALLED: 
            CDDMConfig.fftlib = name
        else:
            warnings.warn("MKL FFT is not installed so it can not be used! Please install mkl_fft.")            
    elif name == "scipy.fftpack" or name == "scipy":
        if SCIPY_INSTALLED:
            CDDMConfig.fftlib = "scipy"
        else:
            warnings.warn("Scipy is not installed so it can not be used! Please install scipy.") 
    elif name == "numpy.fft" or name == "numpy":
        CDDMConfig.fftlib = "numpy"
    else:
        raise ValueError("Unsupported fft library!")
    return out    

def set_rfft2lib(name = "numpy.fft"):
    """Sets fft library. Returns previous setting."""
    out, name = CDDMConfig.rfft2lib, str(name) 
    if name == "mkl_fft":
        if MKL_FFT_INSTALLED: 
            CDDMConfig.rfft2lib = name
        else:
            warnings.warn("MKL FFT is not installed so it can not be used! Please install mkl_fft.")            
    elif name == "scipy.fftpack" or name == "scipy":
        if SCIPY_INSTALLED:
            CDDMConfig.rfft2lib = "scipy"
        else:
            warnings.warn("Scipy is not installed so it can not be used! Please install scipy.") 
    elif name == "numpy.fft" or name == "numpy":
        CDDMConfig.rfft2lib = "numpy"
    else:
        raise ValueError("Unsupported fft library!")
    return out    

set_fftlib(_readconfig(config.get, "fft", "fftlib", ("mkl_fft" if MKL_FFT_INSTALLED else "numpy")))
set_rfft2lib(_readconfig(config.get, "fft", "rfft2lib", "numpy"))
set_verbose(_readconfig(config.getint, "core", "verbose", 0))
set_cv2(_readconfig(config.getboolean, "core", "cv2", True if CV2_INSTALLED else False))

import numba

F32 = numba.float32
F64 = numba.float64
C64 = numba.complex64
C128 = numba.complex128
U8 = numba.uint8
U16 = numba.uint16
U32 = numba.uint32
U64 = numba.uint64

I32 = numba.int32
I64 = numba.int64

if _read_environ_variable("CDDM_DOUBLE_PRECISION",
  default =  _readconfig(config.getboolean, "core", "double_precision", True)):
    F = F64
    C = C128
    U = U64
    I = I64
else:
    F = F32
    C = C64
    U = U32
    I = I32
  
    
