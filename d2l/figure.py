"""The image module contains functions for plotting"""
from IPython import display
from matplotlib import pyplot as plt
from mxnet import nd
from .d2l import set_figsize, use_svg_display
import numpy as np


def _make_list(obj, default_values=None):
    if obj is None:
        obj = default_values
    elif not isinstance(obj, (list, tuple)):
        obj = [obj]
    return obj
