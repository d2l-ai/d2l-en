from ..figure import use_svg_display, plt, plot, show
from .train import evaluate_accuracy
from mxnet import autograd

__all__  = ['sgd', 'squared_loss', 'train_ch3']

def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()

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

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    trains, test_accs = [], []
    use_svg_display()
    fig, ax = plt.subplots(figsize=(4,3))
    for epoch in range(num_epochs):
        trains.append(train_epoch_ch3(net, train_iter, loss, updater))
        test_accs.append(evaluate_accuracy(test_iter, net))
        legend = ['train loss', 'train acc', 'test acc']
        res = list(map(list, zip(*trains)))+[test_accs,]
        plot(list(range(1, epoch+2)), res, 'epoch', legend=legend,
             xlim=[0,num_epochs+1], ylim=[0.4, 0.9], axes=ax)
        show(fig)
