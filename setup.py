#!/usr/bin/env python
from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
import numpy


with open('README.rst') as file:
    long_description = file.read()

kwargs = {'libraries': [], 'include_dirs': [numpy.get_include()]}

ext_modules = [Extension("bekk.recursion",
                         ["./bekk/recursion.pyx"], **kwargs),
               Extension("bekk.likelihood",
                         ["./bekk/likelihood.pyx"], **kwargs)]

setup(name='bekk',
      version='1.0',
      description='Simulation and estimation of BEKK(1,1) model',
      long_description=long_description,
      author='Stanislav Khrapov',
      author_email='khrapovs@gmail.com',
      url='https://github.com/khrapovs/bekk',
      license='MIT',
      keywords=['BEKK', 'ARCH', 'GARCH', 'multivariate', 'volatility'],
      install_requires=['numpy', 'cython', 'scipy', 'pandas',
                        'pandas_datareader'],
      packages=find_packages(),
      ext_modules=ext_modules,
      package_dir={'bekk': './bekk'},
      cmdclass={'build_ext': build_ext},
      include_dirs=[numpy.get_include()],
      zip_safe=False,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
      ],
      )
