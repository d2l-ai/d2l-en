import collections
import math
import os
import random
import sys
import tarfile
import time
import zipfile

import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
import numpy as np

# __all__ = ['VOC_CLASSES', 'Benchmark']
