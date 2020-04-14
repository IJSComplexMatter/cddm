"""Demonstrates how to create mask array for data masking during computation"""

from cddm.map import k_indexmap, plot_indexmap

kmap = k_indexmap(63,32, angle = 0, sector = 180)
mask = (kmap > 20) & (kmap < 30)

if __name__ == "__main__":
    plot_indexmap(mask)