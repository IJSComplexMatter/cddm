"""Plots random walk for first 6 particles"""
#change CWD to this file's path
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from cddm.sim import plot_random_walk, seed
from conf import NFRAMES, SIMSHAPE
seed(0)

plot_random_walk(count = NFRAMES, particles = 6, shape = SIMSHAPE)
