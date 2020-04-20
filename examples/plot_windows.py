#change CWD to this file's path
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from cddm.window import plot_windows
plot_windows()