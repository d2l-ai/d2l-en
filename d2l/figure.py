"""The image module contains functions for plotting"""
from IPython import display
from matplotlib import pyplot as plt
from mxnet import nd
from .d2l import set_figsize, use_svg_display
import numpy as np

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0],
                         height=bbox[3]-bbox[1], fill=False, edgecolor=color,
                         linewidth=2)

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """Plot x and log(y)."""
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

def _preprocess_2d(X):
    def _nd2np(x):
        return x.asnumpy() if isinstance(x, nd.NDArray) else x
    X = _nd2np(X)
    if isinstance(X, list) or isinstance(X, tuple):
        X = np.array([_nd2np(x) for x in X])
    if X.ndim == 1:
        X = X.reshape((1, -1))
    return X

def _check_shape_2d(X, Y):
    assert X.ndim == 2, ('X', X)
    assert Y.ndim == 2, ('Y', Y)
    assert X.shape[-1] == Y.shape[-1], ('X', X, 'Y', Y)
    assert len(X) == len(Y) or len(X) == 1, ('X', X, 'Y', Y)

def _make_list(obj, default_values=None):
    if obj is None:
        obj = default_values
    elif not isinstance(obj, (list, tuple)):
        obj = [obj]
    return obj

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'k'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
