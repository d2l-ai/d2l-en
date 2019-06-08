import sys
d2l = sys.modules[__name__]

# Defined in file: ./chapter_preface/preface.md
from IPython import display
import os
import sys
from matplotlib import pyplot as plt
from mxnet import nd, autograd, gluon, init, context
from mxnet.gluon import nn
import time

# Defined in file: ./chapter_crashcourse/probability.md
def use_svg_display():
    """Use the svg format to display plot in jupyter."""
    display.set_matplotlib_formats('svg')

# Defined in file: ./chapter_crashcourse/naive-bayes.md
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    axes = plt.subplots(num_rows, num_cols, figsize=figsize)[1].flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def synthetic_data(w, b, num_examples):
    """generate y = X w + b + noise"""
    X = nd.random.normal(scale=1, shape=(num_examples, len(w)))
    y = nd.dot(X, w) + b
    y += nd.random.normal(scale=0.01, shape=y.shape)
    return X, y

# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def linreg(X, w, b):
    return nd.dot(X, w) + b

# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

# Defined in file: ./chapter_linear-networks/linear-regression-gluon.md
def load_array(features, labels, batch_size, is_train=True):
    """Construct a Gluon data loader"""
    dataset = gluon.data.ArrayDataset(features, labels)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
    

# Defined in file: ./chapter_linear-networks/fashion-mnist.md
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# Defined in file: ./chapter_linear-networks/fashion-mnist.md
def get_dataloader_workers(num_workers=4):
    # 0 means no additional process is used to speed up the reading of data.
    if sys.platform.startswith('win'):
        return 0
    else:
        return num_workers

# Defined in file: ./chapter_linear-networks/fashion-mnist.md
def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.Resize(resize)] if resize else []
    trans.append(dataset.transforms.ToTensor())
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))

# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()

# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def evaluate_accuracy(net, data_iter):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += accuracy(net(X), y)
        n += y.size
    return acc_sum / n

# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def train_epoch_ch3(net, train_iter, loss, updater):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        # compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater()
        # measure loss and accuracy
        train_l_sum += l.sum().asscalar()
        train_acc_sum += accuracy(y_hat, y)
        n += y.size
    return train_l_sum/n, train_acc_sum/n

# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def plot(X, Y, x_label=None, y_label=None, legend=None, 
         xlim=None, ylim=None, axes=None):
    """Plot multiple lines"""
    axes = axes if axes else d2l.plt.gca()
    draw(axes, axes.plot, X, Y, x_label, y_label, legend, xlim, ylim)
    

# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def draw(axes, func, X, Y, x_label, y_label, legend, xlim, ylim):
    """Draw multiple data series with customized func"""
    if not hasattr(X[0], "__len__") or len(X[0]) != len(Y[0]):
        X = [X] * len(Y)
    axes.cla()
    for x, y in zip(X, Y):
        func(x, y)
    if x_label: axes.set_xlabel(x_label)
    if y_label: axes.set_ylabel(y_label)
    if xlim: axes.set_xlim(xlim)
    if ylim: axes.set_ylim(ylim)
    if legend: axes.legend(legend)

# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def show(obj):
    """Show a figure"""
    display.display(obj)
    display.clear_output(wait=True)

# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    trains, test_accs = [], []
    d2l.use_svg_display()
    fig, ax = d2l.plt.subplots(figsize=(4,3))
    for epoch in range(num_epochs):
        trains.append(train_epoch_ch3(net, train_iter, loss, updater))
        test_accs.append(evaluate_accuracy(net, test_iter))
        legend = ['train loss', 'train acc', 'test acc']
        res = list(map(list, zip(*trains)))+[test_accs,]
        plot(list(range(1, epoch+2)), res, 'epoch', legend=legend, 
             xlim=[0,num_epochs+1], ylim=[0.3, 0.9], axes=ax)
        show(fig)

# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def predict_ch3(net, test_iter, n=9):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y.asnumpy())
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
    titles = [true+'\n'+ pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)), 1, n, titles=titles[0:n])
    

# Defined in file: ./chapter_deep-learning-computation/use-gpu.md
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    return context.gpu(i) if context.num_gpus() >= i else context.cpu()

# Defined in file: ./chapter_deep-learning-computation/use-gpu.md
def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    ctxes = [context.gpu(i) for i in range(context.num_gpus())]
    return ctxes if ctxes else [context.cpu()]
        

# Defined in file: ./chapter_convolutional-neural-networks/conv-layer.md
def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

# Defined in file: ./chapter_convolutional-neural-networks/lenet.md
def evaluate_accuracy(net, data_iter):
    # Query on which device the parameter is.
    ctx = list(net.collect_params().values())[0].list_ctx()[0]
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n

# Defined in file: ./chapter_convolutional-neural-networks/lenet.md
def train_epoch_ch5(net, train_iter, batch_size, loss, trainer, ctx):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        # Here is the only difference compared to train_epoch_ch3
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        trainer.step(batch_size)
        train_l_sum += l.sum().asscalar()
        train_acc_sum += d2l.accuracy(y_hat, y)
        n += y.size
    return train_l_sum / n, train_acc_sum / n

# Defined in file: ./chapter_convolutional-neural-networks/lenet.md
def train_ch5(net, train_iter, test_iter, batch_size, num_epochs, lr, 
              ctx=d2l.try_gpu()):
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    trains, test_accs = [], []
    d2l.use_svg_display()
    fig, ax = d2l.plt.subplots(figsize=(4, 3))
    start = time.time()
    for epoch in range(num_epochs):
        trains.append(train_epoch_ch5(
            net, train_iter, batch_size, loss, trainer, ctx))
        test_accs.append(evaluate_accuracy(net, test_iter))
        legend = ['train loss', 'train acc', 'test acc']
        res = list(map(list, zip(*trains)))+[test_accs,]
        d2l.plot(list(range(1, epoch+2)), res, 'epoch', legend=legend, 
             xlim=[0,num_epochs+1], ylim=[0, 1], axes=ax)
        d2l.show(fig)
    print('Done in %d sec on %s, loss %.3f, train acc %.3f, test acc %.3f'%(
        time.time()-start, ctx, *trains[-1], test_accs[-1]))

# Defined in file: ./chapter_generative_adversarial_networks/gan.md
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator"""
    batch_size = X.shape[0]
    ones = nd.ones((batch_size,), ctx=X.context)
    zeros = nd.zeros((batch_size,), ctx=X.context)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # Don't need to compute gradient for net_G, detach it from
        # computing gradients.
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return loss_D.mean().asscalar()

# Defined in file: ./chapter_generative_adversarial_networks/gan.md
def update_G(Z, net_D, net_G, loss, trainer_G):  # saved in d2l
    """Update generator"""
    batch_size = Z.shape[0]
    ones = nd.ones((batch_size,), ctx=Z.context)
    with autograd.record():
        # We could reuse fake_X from update_D to save computation.
        fake_X = net_G(Z)
        # Recomputing fake_Y is needed since net_D is changed.
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return loss_G.mean().asscalar()

# Defined in file: ./chapter_generative_adversarial_networks/gan.md
def scatter(X, Y, x_label=None, y_label=None, legend=None,
            xlim=None, ylim=None, axes=None):
    """A scatter plot of multiple data series"""
    axes = axes if axes else d2l.plt.gca()
    d2l.draw(axes, axes.scatter, X, Y, x_label, y_label, legend, xlim, ylim)

