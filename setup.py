#!/usr/bin/env python
from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
import cython_gsl
import numpy


with open('README.rst') as file:
    long_description = file.read()

ckwargs = {'libraries': cython_gsl.get_libraries(),
           'library_dirs': [cython_gsl.get_library_dir()],
           'include_dirs': [cython_gsl.get_cython_include_dir()]}

ext_modules = [Extension("bekk.recursion",
                         ["./bekk/recursion.pyx"], **ckwargs),
               Extension("bekk.likelihood",
                         ["./bekk/likelihood.pyx"], **ckwargs)]

setup(name='bekk',
      version='1.0',
      description=('Simulation and estimation of BEKK(1,1) model'),
      long_description=long_description,
      author='Stanislav Khrapov',
      license='MIT',
      author_email='khrapovs@gmail.com',
      url='https://github.com/khrapovs/bekk',
      py_modules=['bekk'],
      package_dir={'bekk': './bekk'},
      packages=find_packages(),
      keywords=['BEKK', 'ARCH', 'GARCH', 'multivariate', 'volatility'],
      include_dirs=[cython_gsl.get_include(), numpy.get_include()],
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules,
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
