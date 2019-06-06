"""The image module contains functions for plotting"""
from IPython import display
from matplotlib import pyplot as plt
from mxnet import nd
import numpy as np

__all__ = ['plt', 'plot', 'scatter', 'show', 'bbox_to_rect', 'semilogy', 'set_figsize', 'show_bboxes',
           'show_images', 'show_trace_2d', 'use_svg_display']

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

def _draw(func, X, Y, x_label, y_label, legend, xlim, ylim, axes):
    use_svg_display()
    X, Y = _preprocess_2d(X), _preprocess_2d(Y)
    _check_shape_2d(X, Y)
    ax = axes if axes else plt.gca()
    ax.cla()
    for i in range(len(Y)):
        x = X[0,:] if len(X) == 1 else X[i,:]
        getattr(ax, func)(x, Y[i,:])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.legend(legend)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def plot(X, Y, x_label=None, y_label=None, legend=None, xlim=None, ylim=None,
         axes=None):
    """plot(x,y) for (x,y) in zip(X,Y)."""
    _draw('plot', X, Y, x_label, y_label, legend, xlim, ylim, axes)

def scatter(X, Y, x_label=None, y_label=None, legend=None, xlim=None, ylim=None,
            axes=None):
    """plot(x,y) for (x,y) in zip(X,Y)."""
    _draw('scatter', X, Y, x_label, y_label, legend, xlim, ylim, axes)


def show(obj):
    display.display(obj)
    display.clear_output(wait=True)

def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

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

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    use_svg_display()
    figsize = (num_cols * scale, num_rows * scale)
    axes = plt.subplots(num_rows, num_cols, figsize=figsize)[1].flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def show_trace_2d(f, res):
    """Show the trace of 2d variables during optimization."""
    x1, x2 = zip(*res)
    set_figsize()
    plt.plot(x1, x2, '-o', color='#ff7f0e')
    x1 = np.arange(-5.5, 1.0, 0.1)
    x2 = np.arange(min(-3.0, min(x2) - 1), max(1.0, max(x2) + 1), 0.1)
    x1, x2 = np.meshgrid(x1, x2)
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')

def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')
