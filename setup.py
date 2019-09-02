from setuptools import setup, find_packages

packages = find_packages()

setup(name = 'cddm',
      version = "0.0.1.dev0",
      description = 'Tools for cross-differential dynamic microscopy',
      author = 'Andrej Petelin',
      author_email = 'andrej.petelin@gmail.com',
      packages = packages,
      #include_package_data=True
    package_data={
        # If any package contains *.dat, or *.ini include them:
        '': ['*.dat',"*.ini"]}
      )