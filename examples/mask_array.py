"""Demonstrates how to create mask array for data masking during computation"""
#change CWD to this file's path
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


from cddm.map import k_indexmap, plot_indexmap

from conf import KISIZE, KJSIZE

kmap = k_indexmap(KISIZE, KJSIZE, angle = 0, sector = 90)
mask = (kmap >= 20) & (kmap <= 30)

if __name__ == "__main__":
    plot_indexmap(mask)