
The following additional libraries are needed to run this
notebook. Note that running on Colab is experimental, please report a Github
issue if you have any problem.


```python
!pip install -U mxnet-cu101mkl==1.6.0  # updating mxnet to at least v1.6
!pip install d2l==0.13.2 -f https://d2l.ai/whl.html # installing d2l

```

    Collecting mxnet-cu101mkl==1.6.0
    [?25l  Downloading https://files.pythonhosted.org/packages/3d/4b/e51dc49ca5fe6564028e7c91b10a3f79c00d710dd691b408c77597df5883/mxnet_cu101mkl-1.6.0-py2.py3-none-manylinux1_x86_64.whl (711.0MB)
    [K     |████████████████████████████████| 711.0MB 24kB/s 
    [?25hCollecting graphviz<0.9.0,>=0.8.1
      Downloading https://files.pythonhosted.org/packages/53/39/4ab213673844e0c004bed8a0781a0721a3f6bb23eb8854ee75c236428892/graphviz-0.8.4-py2.py3-none-any.whl
    Requirement already satisfied, skipping upgrade: numpy<2.0.0,>1.16.0 in /usr/local/lib/python3.6/dist-packages (from mxnet-cu101mkl==1.6.0) (1.18.4)
    Requirement already satisfied, skipping upgrade: requests<3,>=2.20.0 in /usr/local/lib/python3.6/dist-packages (from mxnet-cu101mkl==1.6.0) (2.23.0)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.20.0->mxnet-cu101mkl==1.6.0) (1.24.3)
    Requirement already satisfied, skipping upgrade: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.20.0->mxnet-cu101mkl==1.6.0) (3.0.4)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.20.0->mxnet-cu101mkl==1.6.0) (2020.4.5.1)
    Requirement already satisfied, skipping upgrade: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.20.0->mxnet-cu101mkl==1.6.0) (2.9)
    Installing collected packages: graphviz, mxnet-cu101mkl
      Found existing installation: graphviz 0.10.1
        Uninstalling graphviz-0.10.1:
          Successfully uninstalled graphviz-0.10.1
    Successfully installed graphviz-0.8.4 mxnet-cu101mkl-1.6.0
    Looking in links: https://d2l.ai/whl.html
    Collecting d2l==0.13.2
      Downloading https://d2l.ai/dist/d2l-0.13.2-py3-none-any.whl
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from d2l==0.13.2) (3.2.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from d2l==0.13.2) (1.18.4)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from d2l==0.13.2) (1.0.4)
    Requirement already satisfied: jupyter in /usr/local/lib/python3.6/dist-packages (from d2l==0.13.2) (1.0.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->d2l==0.13.2) (1.2.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->d2l==0.13.2) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->d2l==0.13.2) (2.4.7)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->d2l==0.13.2) (0.10.0)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas->d2l==0.13.2) (2018.9)
    Requirement already satisfied: ipykernel in /usr/local/lib/python3.6/dist-packages (from jupyter->d2l==0.13.2) (4.10.1)
    Requirement already satisfied: ipywidgets in /usr/local/lib/python3.6/dist-packages (from jupyter->d2l==0.13.2) (7.5.1)
    Requirement already satisfied: nbconvert in /usr/local/lib/python3.6/dist-packages (from jupyter->d2l==0.13.2) (5.6.1)
    Requirement already satisfied: qtconsole in /usr/local/lib/python3.6/dist-packages (from jupyter->d2l==0.13.2) (4.7.4)
    Requirement already satisfied: notebook in /usr/local/lib/python3.6/dist-packages (from jupyter->d2l==0.13.2) (5.2.2)
    Requirement already satisfied: jupyter-console in /usr/local/lib/python3.6/dist-packages (from jupyter->d2l==0.13.2) (5.2.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil>=2.1->matplotlib->d2l==0.13.2) (1.12.0)
    Requirement already satisfied: ipython>=4.0.0 in /usr/local/lib/python3.6/dist-packages (from ipykernel->jupyter->d2l==0.13.2) (5.5.0)
    Requirement already satisfied: jupyter-client in /usr/local/lib/python3.6/dist-packages (from ipykernel->jupyter->d2l==0.13.2) (5.3.4)
    Requirement already satisfied: traitlets>=4.1.0 in /usr/local/lib/python3.6/dist-packages (from ipykernel->jupyter->d2l==0.13.2) (4.3.3)
    Requirement already satisfied: tornado>=4.0 in /usr/local/lib/python3.6/dist-packages (from ipykernel->jupyter->d2l==0.13.2) (4.5.3)
    Requirement already satisfied: nbformat>=4.2.0 in /usr/local/lib/python3.6/dist-packages (from ipywidgets->jupyter->d2l==0.13.2) (5.0.6)
    Requirement already satisfied: widgetsnbextension~=3.5.0 in /usr/local/lib/python3.6/dist-packages (from ipywidgets->jupyter->d2l==0.13.2) (3.5.1)
    Requirement already satisfied: testpath in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->d2l==0.13.2) (0.4.4)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->d2l==0.13.2) (1.4.2)
    Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->d2l==0.13.2) (0.3)
    Requirement already satisfied: jinja2>=2.4 in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->d2l==0.13.2) (2.11.2)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->d2l==0.13.2) (0.6.0)
    Requirement already satisfied: jupyter-core in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->d2l==0.13.2) (4.6.3)
    Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->d2l==0.13.2) (0.8.4)
    Requirement already satisfied: bleach in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->d2l==0.13.2) (3.1.5)
    Requirement already satisfied: pygments in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->d2l==0.13.2) (2.1.3)
    Requirement already satisfied: pyzmq>=17.1 in /usr/local/lib/python3.6/dist-packages (from qtconsole->jupyter->d2l==0.13.2) (19.0.1)
    Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.6/dist-packages (from qtconsole->jupyter->d2l==0.13.2) (0.2.0)
    Requirement already satisfied: qtpy in /usr/local/lib/python3.6/dist-packages (from qtconsole->jupyter->d2l==0.13.2) (1.9.0)
    Requirement already satisfied: terminado>=0.3.3; sys_platform != "win32" in /usr/local/lib/python3.6/dist-packages (from notebook->jupyter->d2l==0.13.2) (0.8.3)
    Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.0 in /usr/local/lib/python3.6/dist-packages (from jupyter-console->jupyter->d2l==0.13.2) (1.0.18)
    Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0->ipykernel->jupyter->d2l==0.13.2) (0.8.1)
    Requirement already satisfied: pexpect; sys_platform != "win32" in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0->ipykernel->jupyter->d2l==0.13.2) (4.8.0)
    Requirement already satisfied: decorator in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0->ipykernel->jupyter->d2l==0.13.2) (4.4.2)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0->ipykernel->jupyter->d2l==0.13.2) (0.7.5)
    Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0->ipykernel->jupyter->d2l==0.13.2) (47.1.1)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /usr/local/lib/python3.6/dist-packages (from nbformat>=4.2.0->ipywidgets->jupyter->d2l==0.13.2) (2.6.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from jinja2>=2.4->nbconvert->jupyter->d2l==0.13.2) (1.1.1)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from bleach->nbconvert->jupyter->d2l==0.13.2) (20.4)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.6/dist-packages (from bleach->nbconvert->jupyter->d2l==0.13.2) (0.5.1)
    Requirement already satisfied: ptyprocess; os_name != "nt" in /usr/local/lib/python3.6/dist-packages (from terminado>=0.3.3; sys_platform != "win32"->notebook->jupyter->d2l==0.13.2) (0.6.0)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.6/dist-packages (from prompt-toolkit<2.0.0,>=1.0.0->jupyter-console->jupyter->d2l==0.13.2) (0.2.2)
    Installing collected packages: d2l
    Successfully installed d2l-0.13.2
    

# Concise Implementation of Softmax Regression
:label:`sec_softmax_gluon`

Just as Gluon made it much easier
to implement linear regression in :numref:`sec_linear_gluon`,
we will find it similarly (or possibly more)
convenient for implementing classification models.
Again, we begin with our import ritual.


```python
from d2l import mxnet as d2l
import tensorflow as tf
```

Let us stick with the Fashion-MNIST dataset
and keep the batch size at $256$ as in the last section.


```python
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
batch_size = 256
```


```python
# Rescaling
train_images = train_images / 255.0

test_images = test_images / 255.0

```


```python
# inspecting the Examples in training and test set
print('No. of Images in training set is {} and in test set {}'.format(train_images.shape[0], test_images.shape[0]))
print('No. of lables in training set is {} and test set is {}'.format(train_labels.shape[0], test_labels.shape[0]))
```

    No. of Images in training set is 60000 and in test set 10000
    No. of lables in training set is 60000 and test set is 10000
    

## Initializing Model Parameters

As mentioned in :numref:`sec_softmax`,
the output layer of softmax regression
is a fully-connected layer.
Therefore, to implement our model,
we just need to add one fully-connected layer
with 10 outputs to our `Sequential`.
Again, here, the `Sequential` is not really necessary,
but we might as well form the habit since it will be ubiquitous
when implementing deep models.
Again, we initialize the weights at random
with zero mean and standard deviation $0.01$.


```python
net = tf.keras.models.Sequential()
# we've to flatten the input images.
net.add(tf.keras.layers.Flatten(input_shape=(28,28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10,kernel_initializer= weight_initializer))
```

## The Softmax

In the previous example, we calculated our model's output
and then ran this output through the cross-entropy loss.
Mathematically, that is a perfectly reasonable thing to do.
However, from a computational perspective,
exponentiation can be a source of numerical stability issues
(as discussed  in :numref:`sec_naive_bayes`).
Recall that the softmax function calculates
$\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}$,
where $\hat y_j$ is the $j^\mathrm{th}$ element of ``y_hat``
and $z_j$ is the $j^\mathrm{th}$ element of the input
``y_linear`` variable, as computed by the softmax.

If some of the $z_i$ are very large (i.e., very positive),
then $e^{z_i}$ might be larger than the largest number
we can have for certain types of ``float`` (i.e., overflow).
This would make the denominator (and/or numerator) ``inf``
and we wind up encountering either $0$, ``inf``, or ``nan`` for $\hat y_j$.
In these situations we do not get a well-defined
return value for ``cross_entropy``.
One trick to get around this is to first subtract $\text{max}(z_i)$
from all $z_i$ before proceeding with the ``softmax`` calculation.
You can verify that this shifting of each $z_i$ by constant factor
does not change the return value of ``softmax``.

After the subtraction and normalization step,
it might be that possible that some $z_j$ have large negative values
and thus that the corresponding $e^{z_j}$ will take values close to zero.
These might be rounded to zero due to finite precision (i.e underflow),
making $\hat y_j$ zero and giving us ``-inf`` for $\text{log}(\hat y_j)$.
A few steps down the road in backpropagation,
we might find ourselves faced with a screenful
of the dreaded not-a-number (``nan``) results.

Fortunately, we are saved by the fact that
even though we are computing exponential functions,
we ultimately intend to take their log
(when calculating the cross-entropy loss).
By combining these two operators
(``softmax`` and ``cross_entropy``) together,
we can escape the numerical stability issues
that might otherwise plague us during backpropagation.
As shown in the equation below, we avoided calculating $e^{z_j}$
and can instead $z_j$ directly due to the canceling in $\log(\exp(\cdot))$.

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}\right) \\
& = \log{(e^{z_j})}-\text{log}{\left( \sum_{i=1}^{n} e^{z_i} \right)} \\
& = z_j -\log{\left( \sum_{i=1}^{n} e^{z_i} \right)}.
\end{aligned}
$$

We will want to keep the conventional softmax function handy
in case we ever want to evaluate the probabilities output by our model.
But instead of passing softmax probabilities into our new loss function,
we will just pass the logits and compute the softmax and its log
all at once inside the softmax_cross_entropy loss function,
which does smart things like the log-sum-exp trick ([see on Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)).


```python
# adding softmax layer 
net.add(tf.keras.layers.Softmax())
# Displaying the architecture of our model
net.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_1 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                7850      
    _________________________________________________________________
    softmax_2 (Softmax)          (None, 10)                0         
    =================================================================
    Total params: 7,850
    Trainable params: 7,850
    Non-trainable params: 0
    _________________________________________________________________
    

## Optimization Algorithm

Here, we use minibatch stochastic gradient descent
with a learning rate of $0.1$ as the optimization algorithm.
Note that this is the same as we applied in the linear regression example
and it illustrates the general applicability of the optimizers.


```python
# Deffining optimizer and loss function for our model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```


```python
# compiling the model
net.compile(optimizer=optimizer, loss= loss, metrics=['accuracy'])
```

## Training

Next we call the training function defined in the last section to train a model.


```python
num_epochs = 10
train_history = net.fit(train_images, train_labels, batch_size=batch_size, epochs=num_epochs)
```

    Epoch 1/10
    235/235 [==============================] - 1s 2ms/step - loss: 1.9843 - accuracy: 0.5723
    Epoch 2/10
    235/235 [==============================] - 1s 2ms/step - loss: 1.8525 - accuracy: 0.6427
    Epoch 3/10
    235/235 [==============================] - 1s 2ms/step - loss: 1.7907 - accuracy: 0.7262
    Epoch 4/10
    235/235 [==============================] - 1s 2ms/step - loss: 1.7591 - accuracy: 0.7536
    Epoch 5/10
    235/235 [==============================] - 1s 2ms/step - loss: 1.7421 - accuracy: 0.7645
    Epoch 6/10
    235/235 [==============================] - 1s 2ms/step - loss: 1.7311 - accuracy: 0.7705
    Epoch 7/10
    235/235 [==============================] - 1s 2ms/step - loss: 1.7231 - accuracy: 0.7744
    Epoch 8/10
    235/235 [==============================] - 1s 2ms/step - loss: 1.7170 - accuracy: 0.7775
    Epoch 9/10
    235/235 [==============================] - 1s 2ms/step - loss: 1.7120 - accuracy: 0.7801
    Epoch 10/10
    235/235 [==============================] - 1s 2ms/step - loss: 1.7079 - accuracy: 0.7828
    


```python
test_history = net.evaluate(test_images, test_labels)
```

    313/313 [==============================] - 0s 1ms/step - loss: 1.7126 - accuracy: 0.7775
    


```python
# Plotting 
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training'], loc='lower right')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training'], loc='upper right')
plt.show() 
```


![png](softmax_regression_concise%28TF%29_files/softmax_regression_concise%28TF%29_18_0.png)


As before, this algorithm converges to a solution
that achieves an accuracy of 83.7%,
albeit this time with fewer lines of code than before.
Note that in many cases, Gluon takes additional precautions
beyond these most well-known tricks to ensure numerical stability,
saving us from even more pitfalls that we would encounter
if we tried to code all of our models from scratch in practice.

## Exercises

1. Try adjusting the hyper-parameters, such as batch size, epoch, and learning rate, to see what the results are.
1. Why might the test accuracy decrease again after a while? How could we fix this?


[Discussions](https://discuss.d2l.ai/t/52)
