from setuptools import setup, find_packages
from cddm import __version__


packages = find_packages()

setup(name = 'cddm',
      version = __version__,
      description = 'Tools for cross-differential dynamic microscopy',
      author = 'Andrej Petelin',
      author_email = 'andrej.petelin@gmail.com',
      packages = packages,
      #include_package_data=True
    package_data={
        # If any package contains *.dat, or *.ini include them:
        '': ['*.dat',"*.ini"]}
      )