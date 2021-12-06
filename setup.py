"""
Source Code from Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

#setup function to compile the cython modules
setup(
    name='SixdSense',
    version='1.0.0',
    description='6d pose detection with NN learned on synthetic datasets',
    ext_modules=cythonize('sds/utils/compute_overlap.pyx'),
    include_dirs=[np.get_include()]
)
