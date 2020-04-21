from setuptools import setup, find_packages
from cddm import __version__

long_description = """
cddm is a python package for analysis of cross-differential dynamic microscopy (cross-DDM) experiments. You can also use it for a single-camera (DDM) analysis, or a general purpose correlation analysis. Regular and irregular time-spaced data is supported. The package is hosted at GitHub
"""

packages = find_packages()

setup(name = 'cddm',
      version = __version__,
      description = 'Tools for cross-differential dynamic microscopy',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author = 'Andrej Petelin',
      author_email = 'andrej.petelin@gmail.com',
      url="https://github.com/IJSComplexMatter/cddm",
      packages = packages,
      #include_package_data=True
      package_data={
        # If any package contains *.dat, or *.ini include them:
        '': ['*.dat',"*.ini"]},
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
      )


