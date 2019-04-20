"""The train module contains functions for neural network training"""
import numpy as np
import math
import time

import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
from .data import data_iter_consecutive, data_iter_random
from .base import try_gpu
from .figure import set_figsize, plt
from .model import linreg

__all__ = ['evaluate_accuracy', 'squared_loss', 'grad_clipping', 'grad_clipping_gluon', 'sgd', 'train',
           'train_2d', 'train_and_predict_rnn', 'train_and_predict_rnn_gluon',
           'train_ch3', 'train_ch5', 'train_ch9', 'train_gluon_ch9',
           'predict_sentiment', 'train_ch7', 'translate_ch7']

def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def grad_clipping(params, theta, ctx):
    """Clip the gradient."""
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def grad_clipping_gluon(model, theta, ctx):
    """Clip the gradient for a Gluon model."""
    params = [p.data(ctx) for p in model.collect_params().values()]
    grad_clipping(params, theta, ctx)

def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    """Train and evaluate a model."""
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                  for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc,
                 time.time() - start))


def train_2d(trainer):
    """Optimize the objective function of 2D variables with a customized trainer."""
    x1, x2 = -5, -2
    s_x1, s_x2 = 0, 0
    res = [(x1, x2)]
    for i in range(20):
        x1, x2, s_x1, s_x2 = trainer(x1, x2, s_x1, s_x2)
        res.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i+1, x1, x2))
    return res

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          corpus_indices, vocab, ctx, is_random_iter,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, prefixes):
    """Train an RNN model and predict the next item in the sequence."""
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()
    start = time.time()
    for epoch in range(1, num_epochs+1):
        if not is_random_iter:
            # If adjacent sampling is used, the hidden state is initialized
            # at the beginning of the epoch
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n = 0.0, 0
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:
                # If random sampling is used, the hidden state is initialized
                # before each mini-batch update
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                # Otherwise, the detach function needs to be used to separate
                # the hidden state from the computational graph to avoid
                # backpropagation beyond the current sample
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, len(vocab))
                # outputs is num_steps terms of shape (batch_size, len(vocab))
                (outputs, state) = rnn(inputs, state, params)
                # After stitching it is (num_steps * batch_size, len(vocab))
                outputs = nd.concat(*outputs, dim=0)
                # The shape of Y is (batch_size, num_steps), and then becomes
                # a vector with a length of batch * num_steps after
                # transposition. This gives it a one-to-one correspondence
                # with output rows
                y = Y.T.reshape((-1,))
                # Average classification error via cross entropy loss
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)  # Clip the gradient
            sgd(params, lr, 1)
            # Since the error is the mean, no need to average gradients here
            l_sum += l.asscalar() * y.size
            n += y.size
        if epoch % (num_epochs // 4) == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch, math.exp(l_sum / n), time.time() - start))
            start = time.time()
        if epoch % (num_epochs // 2) == 0:
            for prefix in prefixes:
                print(' -',  predict_rnn(prefix, 50, rnn, params,
                                         init_rnn_state, num_hiddens,
                                         vocab, ctx))


def train_and_predict_rnn_gluon(model, num_hiddens, corpus_indices, vocab,
                                ctx, num_epochs, num_steps, lr,
                                clipping_theta, batch_size, prefixes):
    """Train a Gluon RNN model and predict the next item in the sequence."""
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})
    start = time.time()
    for epoch in range(1, num_epochs+1):
        l_sum, n = 0.0, 0
        data_iter = data_iter_consecutive(
            corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            # Clip the gradient
            grad_clipping_gluon(model, clipping_theta, ctx)
            # Since the error has already taken the mean, the gradient does
            # not need to be averaged
            trainer.step(1)
            l_sum += l.asscalar() * y.size
            n += y.size

        if epoch % (num_epochs // 4) == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch, math.exp(l_sum / n), time.time() - start))
            start = time.time()
        if epoch % (num_epochs // 2) == 0:
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(prefix, 50, model, vocab, ctx))


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    """Train and evaluate a model with CPU."""
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))


def train_ch9(trainer_fn, states, hyperparams, features, labels, batch_size=10,
              num_epochs=2):
    """Train a linear regression model."""
    net, loss = linreg, squared_loss
    w, b = nd.random.normal(scale=0.01, shape=(
        features.shape[1], 1)), nd.zeros(1)
    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features, w, b), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')


def train_gluon_ch9(trainer_name, trainer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    """Train a linear regression model with a given Gluon trainer."""
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()

    def eval_loss():
        return loss(net(features), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(),
                            trainer_name, trainer_hyperparams)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')

def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab, ctx):
    """Predict next chars with an RNN model"""
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [vocab[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # The output of the previous time step is taken as the input of the
        # current time step
        X = to_onehot(nd.array([output[-1]], ctx=ctx), len(vocab))
        # Calculate the output and update the hidden state
        (Y, state) = rnn(X, state, params)
        # The input to the next time step is the character in the prefix or
        # the current best predicted character
        if t < len(prefix) - 1:
            # Read off from the given sequence of characters
            output.append(vocab[prefix[t + 1]])
        else:
            # This is maximum likelihood decoding. Modify this if you want to
            # use sampling, beam search or beam sampling for better sequences.
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([vocab.idx_to_token[i] for i in output])


def predict_rnn_gluon(prefix, num_chars, model, vocab, ctx):
    """Predict next chars with a Gluon RNN model."""
    # Use the model's member function to initialize the hidden state
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [vocab[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        # Forward computation does not require incoming model parameters
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(vocab[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([vocab.idx_to_token[i] for i in output])

def predict_sentiment(net, vocab, sentence):
    """Predict the sentiment of a given sentence."""
    sentence = nd.array(vocab[sentence.split()], ctx=try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'

def train_ch7(model, data_iter, lr, num_epochs, ctx):
    """Train an encoder-decoder model"""
    model.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),
                            'adam', {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    tic = time.time()
    for epoch in range(1, num_epochs+1):
        l_sum, num_tokens_sum = 0.0, 0.0
        for batch in data_iter:
            X, X_vlen, Y, Y_vlen = [x.as_in_context(ctx) for x in batch]
            Y_input, Y_label, Y_vlen = Y[:,:-1], Y[:,1:], Y_vlen-1
            with autograd.record():
                Y_hat, _ = model(X, Y_input, X_vlen, Y_vlen)
                l = loss(Y_hat, Y_label, Y_vlen)
            l.backward()
            grad_clipping_gluon(model, 5, ctx)
            num_tokens = Y_vlen.sum().asscalar()
            trainer.step(num_tokens)
            l_sum += l.sum().asscalar()
            num_tokens_sum += num_tokens
        if epoch % (num_epochs // 4) == 0:
            print("epoch %d, loss %.3f, time %.1f sec" % (
                epoch, l_sum/num_tokens_sum, time.time()-tic))
            tic = time.time()

def translate_ch7(model, src_sentence, src_vocab, tgt_vocab, max_len, ctx):
    """Translate based on an encoder-decoder model with greedy search."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')]
    src_len = len(src_tokens)
    if src_len < max_len:
        src_tokens += [src_vocab.pad] * (max_len - src_len)
    enc_X = nd.array(src_tokens, ctx=ctx)
    enc_valid_length = nd.array([src_len], ctx=ctx)
    enc_outputs = model.encoder(enc_X.expand_dims(axis=0), enc_valid_length)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_length)
    dec_X = nd.array([tgt_vocab.bos], ctx=ctx).expand_dims(axis=0)
    predict_tokens = []
    for _ in range(max_len):
        Y, dec_state = model.decoder(dec_X, dec_state)
        dec_X = Y.argmax(axis=2)
        py = dec_X.squeeze(axis=0).astype('int32').asscalar()
        if py == tgt_vocab.eos:
            break
        predict_tokens.append(py)
    return ' '.join(tgt_vocab.to_tokens(predict_tokens))

class MaskedSoftmaxCELoss(gloss.SoftmaxCELoss):
    def forward(self, pred, label, valid_length):
        weights = nd.ones_like(label).expand_dims(axis=-1)
        weights = nd.SequenceMask(weights, valid_length, True, axis=1)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)
