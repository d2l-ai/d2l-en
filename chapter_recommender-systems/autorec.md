# AutoRec: An Autoencoder Framework for Rating Prediction

Although the matrix factorization algorithm achieves decent performances on the rating prediction task, it is essentially a linear model. Such models are not capable of capturing complex nonlinear and intricate relationships that may be predictive of users' preferences. In this section, we introduce a nonlinear neural network CF model, Autorec, that identifies CF as an autoencoding architecture. This model aims to integrate nonlinear transformations into CF on the basis of explicit feedback.  Neural networks have been proven to be capable of approximating any continuous function,  making it suitable to address this limitation and enrich the expressiveness of matrix factorization.   

This model has the same structure as autoencoders which consist of an input layer, a hidden layer and a reconstruction (output) layer.  An autoencoder is a neural network that learns to copy its input to its output in order to code the inputs into the hidden, and usually low dimensional representations. In Autorec, instead of explicitly embedding users/items into a low-dimensional space, it uses the column/row of the interaction matrix as input, then reconstructs the interaction matrix in the output layer.


Autorec differs from traditional autoencoder in that autorec focuses on the output layer instead of learning the hidden representations. It uses a partial observed interaction matrix as input, aiming to reconstruct a full rating matrix by predict missing entries in the output layer for the purposes of recommendation.  

There are two variants of Autorec, user and item based Autorec. For the sake of brevity, we only introduce the item based AutoRec. User based AutoRec can be derived accordingly.

Let $\mathbf{R}_{*i}$ denote the $i^{th}$ column of the rating matrix. The unknown ratings  are set to zeros by default. The neural architecture is defined as:

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

where $f(\cdot)$ and $g(\cdot)$ represent activation functions. $\mathbf{W}$ and $\mathbf{V}$ are weight matrices, $\mu$ and $b$ are biases. The output $h(\mathbf{R}_{*i})$ is the reconstruction of the $i^{th}$ column of the rating matrix.

The following objective function aims to minimize the reconstruction error.

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\arg \min} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\parallel \mathbf{W} \parallel_F^2 + \parallel \mathbf{V}\parallel_F^2)
$$

where $\parallel \cdot \parallel_{\mathcal{O}}$ means only the contribution of observed ratings are considered, that is, only weights that are associated with observed inputs are updated during backpropagation.

```{.python .input  n=1}
import d2l
from mxnet import autograd, init, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## Model Implementation

A typical autoencoder consists of an encoder and a decoder. The encoder projects input to hidden representations and the decoder maps the hidden layer to the reconstruction layer. We follow this practice and create the encoder and decoder with dense layers. The activation of encoder is set to `sigmoid` by default and no activation is applied for decoder. A dropout layer is also included after the encoding transformation to prevent the network from overfitting. The gradients of unobserved input are masked out to ensure that only observed ratings contribute to the model learning process.

```{.python .input  n=2}
class autorec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout_rate=0.05):
        super(autorec, self).__init__()
        self.encoder = gluon.nn.Dense(num_hidden, activation='sigmoid', 
                                      use_bias=True)
        self.decoder = gluon.nn.Dense(num_users, use_bias=True)
        self.dropout_layer = gluon.nn.Dropout(dropout_rate)
    def forward(self, input):
        hidden = self.dropout_layer(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training(): # mask the gradient during training.
            return pred * np.sign(input)
        else:
            return pred
```

## Reimplement the Evaluator

Since the input and output have be changed, we need to reimplement the evaluation function but we still use RMSE as the accuracy measure.

```{.python .input  n=3}
def evaluator(network, inter_matrix,  test_data, ctx):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, ctx, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Caculate the test RMSE.
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons)) 
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## Train and Evaluate the Model
Now, let's train and evaluate AutoRec on the MovieLens dataset. We can clearly see that the test RMSE is lower than the matrix factorization model, confirming the effectiveness of neural networks in the rating prediction task.

```{.python .input  n=5}
ctx = d2l.try_all_gpus()
# Load the MovieLens 100K dataset.
df, num_users, num_items = d2l.read_data()
train_data, test_data = d2l.split_data(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_dataset(train_data, num_users, num_items)
_, _, _, test_inter_mat = d2l.load_dataset(test_data, num_users, num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True, 
                                   last_batch="rollover", 
                                   batch_size=128)
test_iter = gluon.data.DataLoader(np.array(train_inter_mat),shuffle=False, 
                                  last_batch="keep", batch_size=1024)
# Model initialization, training and evaluation.
net = autorec(500, num_users)
net.initialize(ctx=ctx, force_reinit=True, init = mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.001, 50, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer, 
                        {"learning_rate": lr, 'wd': wd})
d2l.train_explicit(net, train_iter, test_iter, loss, trainer, num_epochs, 
                   ctx, evaluator, inter_mat = test_inter_mat)
```

## Summary
* We can frame the matrix factorization algorithm with an autoencoder structure, while integrating non-linear layers and dropout reguralization. 
* Experiments on MovieLens 100K dataset show that AutoRec achieves superior performance than matrix factorization.



## Exercises
* Varying the hidden dimension of AutoRec to see its impacts on the model performances.
* Will adding more hidden layers be helpful?
* Change the activation functions and rerun this model, can you find the best combination of decoder and encoder activation functions?

## References
* Sedhain, Suvash, et al. "Autorec: Autoencoders meet collaborative filtering." Proceedings of the 24th International Conference on World Wide Web. ACM, 2015.
