"""Demonstrates how to create mask array for data masking during computation"""
from cddm.map import k_indexmap, plot_indexmap

from examples.conf import KISIZE, KJSIZE

kmap = k_indexmap(KISIZE, KJSIZE, angle = 0, sector = 90)
mask = (kmap >= 20) & (kmap <= 30)

if __name__ == "__main__":
    plot_indexmap(mask)