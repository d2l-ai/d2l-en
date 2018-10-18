# Deep Convolutional Neural Networks (AlexNet)

In nearly two decades since LeNet was proposed, for a time, neural networks were surpassed by other machine learning methods, such as support vector machines. Although LeNet achieved good results on early small data sets, its performance on larger real data sets is not satisfactory. On the one hand, neural network computing is complex. Although some acceleration hardware for neural networks was developed in the 1990s, they were not as popular as GPUs. Therefore, it was difficult to train a multichannel, multilayer convolutional neural network with a large number of parameters in those years. On the other hand, researchers had not yet delved into many research fields, such as parameter initialization, and non-convex optimization algorithms. The lack of such research was another reason why the training of complex neural networks was very difficult. 

As we saw in the previous section, neural networks can classify images directly based on the raw pixels of the image. This method, called end-to-end, greatly reduces the number of intermediate steps. However, the manual features that researchers have designed and generated through hard work and intelligence remained more popular for a long time. The main processes in this sort of image classification research are:

1. obtaining an image data set,
2. generating image features with the existing feature extraction functions, and
3. using machine learning models to classify the features of images.

At the time, machine learning was limited to the final step. If you talked to machine learning researchers at that time, they would have said that machine learning was both important and, in a way, beautiful. Machine learning uses elegant theorems to prove the nature of many classifiers. The machine learning field is vibrant, rigorous, and extremely useful. However, if you talked to computer vision researchers, they would have told you something else. They would tell you that the "hidden" reality in image recognition is that it is data and features that are really important in computer vision processes. That is to say, the use of cleaner data sets and more effective features has a greater impact on the results of image classification than the choice of machine learning models.


## Learning Feature Representation

Because the features are so important, how should they be represented?

As we already mentioned, for a long time, features were extracted from data based on a variety of hand-designed functions. In fact, many researchers continued to improve image classification results by proposing new feature extraction functions. Their efforts made important contributions to the development of computer vision.

However, other researchers disagreed with this approach. They believed that the features themselves should also be obtained from learning. In addition, they held that the features themselves should be represented hierarchically in order to characterize sufficiently complex inputs. Researchers who supported this idea believed that multilayer neural networks may be able to learn multilevel representations of data and represent increasingly abstract concepts or patterns step-by-step. Let us use image classification as an example. Recall the object edge detection example in the ["Two-dimensional Convolutional Layer"](conv-layer.md) section. In multilayer neural networks, the first level representation of an image can be whether or not an edge appears at a particular location and angle. The second level representation may be able to combine these edges into interesting patterns, such as decorative patterns. In the third-level representation, perhaps the decorative pattern of the previous level could be further combined into the pattern of a specific part of the corresponding object. This process is repeated step by step, and finally, the model can easily complete the classification task based on the representation of the last level. It should be emphasized that the hierarchical representation of the input is determined by the parameters in the multilayer model, and these parameters are all obtained from learning.

Although a group of dedicated researchers dedicated themselves to this idea and attempted to study the hierarchical representation of visual data, their ambitions went unrewarded for a long time. It is worthwhile for us to analysis these factors one by one. 


### The First Missing Element: Data

A deep model with many features requires a large amount of labeled data to achieve better results than other typical methods. Constrained by the limited storage of early computers and the limited research budgets of the 1990s, most of the research was based only on small public data sets. For example, many research papers were based on a few public data sets provided by UCI. Many of these data sets had only a few hundred to several thousand images. This situation was improved by the advent of big data around 2010. In particular, the ImageNet data set, which was released in 2009, contains 1,000 categories of objects, each with thousands of different images. This scale was unattainable by the other public data sets at that time. The ImageNet data set has pushed both computer vision and machine learning research into a new phase, so the traditional methods of the past were no longer advantageous.


### The Second Missing Element: Hardware

Deep learning demands massive amounts of computing resources. The computing power of early hardware was limited, making it difficult to train more complex neural networks. However, the release of general-purpose GPUs changed all that. GPUs have long been designed for image processing and computer games, especially for high-throughput matrixes and vector multiplication used for basic graphics conversion. Fortunately, the mathematical expressions used in these application are similar to the expressions of convolutional layers in deep networks. The concept of general-purpose GPUs emerged in 2001, and programming frameworks, such as OpenCL and CUDA, were born. The machine learning community began to use GPUs around 2010.


## AlexNet

 In 2012, AlexNet came into the world. This model was named after Alex Krizhevsky, the first author of the paper proposing the network[1]. AlexNet used an 8-layer convolutional neural network and won the ImageNet Large Scale Visual Recognition Challenge 2012 with a big advantage. This network proved, for the first time, that the features obtained by learning can transcend manually-design features, thus overturning the previous paradigm of computer vision research.

The design philosophies of AlextNet and LeNet are very similar, but there are also significant differences.

First, compared with the relatively small LeNet, AlexNet consists of eight layers, five convolutional layers, two fully connected hidden layers, and one fully connected output layer. Below, we will describe the design of these layers in detail.

In AlexNet's first layer, the convolution window shape is $11\times11$. Since most images in ImageNet are more than ten times higher and wider than the MNIST images, objects in ImageNet images take up more pixels. Consequently, a larger convolution window is needed to capture the object. The convolution window shape in the second layer is reduced to $5\times5$, followed by $3\times3$. In addition, after the first, second, and fifth convolutional layers, the network adds maximum pooling layers with a window shape of $3\times3$ and a stride of 2. Moreover, AlexNet has ten times more convolution channels than LeNet.

After the last convolutional layer are two fully connected layers with 4096 outputs. These two huge fully connected layers produce model parameters of nearly 1 GB. Due to the limited video memory in early GPUs, the original AlexNet used a dual data stream design, so that one GPU only needs to process half of the model. Fortunately, GPU memory has developed tremendously over the past few years, so we usually do not need this special design anymore.

Second, AlextNet changed the sigmoid activation function to a simpler ReLU activation function. On the one hand, the computation of the ReLU activation function is simpler. For example, it does not have the exponentiation operation found in the sigmoid activation function. On the other hand, the ReLU activation function makes model training easier when using different parameter initialization methods. This is because, when the output of the sigmoid activation function is very close to 0 or 1, the gradient of these regions is almost 0, so that back propagation cannot continue to update some of the model parameters. In contrast, the gradient of the ReLU activation function in the positive interval is always 1. Therefore, if the model parameters are not properly initialized, the sigmoid function may obtain a gradient of almost 0 in the positive interval, so that the model cannot be effectively trained.

Third, AlextNet controls the model complexity of the fully connected layer by dropout (see the ["Dropout" ](../chapter_deep-learning-basics/dropout.md) section), while LeNet does not use dropout.

Fourth, AlextNet introduces a great deal of image augmentation, such as flipping, clipping, and color changes, in order to further expand the data set to mitigate overfitting. We will cover this method in detail later in the ["Image Augmentation" ](chapter_computer-vision/image-augmentation.md) section.

Next, we will implement a slightly simplified AlexNet.

```{.python .input  n=1}
import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, nn
import os
import sys

net = nn.Sequential()
# Here, we use a larger 11 x 11 window to capture objects. At the same time, we use a stride of 4 to greatly reduce the height and width of the output.
# Here, the number of input channels is much larger than that in LeNet.
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent height and width across the input and output, and increase the number of output channels
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution window. Except for the final convolutional layer, the number of output channels is further increased.
        # Pooling layers are not used to reduce the height and width of input after the first two convolutional layers.
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Here, the number of outputs of the fully connected layer is several times larger than that in LeNet. Use the dropout layer to mitigate overfitting.
        nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
        nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of classes is 10, instead of 1000 as in the paper.
        nn.Dense(10))
```

We construct a single-channel data instance with both height and width of 224 to observe the output shape of each layer

```{.python .input  n=2}
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

## Reading Data

Although AlexNet uses ImageNet data in the paper, in this example, we again use the Fashion-MNIST data set to demonstrate AlexNet. This is because training with ImageNet data would take too long. When reading the data, we introduced an extra step to expand the image height and width to 224 as used by AlexNet. This can be done with the `Resize` class. That is to say, before using the `ToTensor` class, we use the `Resize` class, and then use the `Compose` class to concatenate these two changes for easy invocation.

```{.python .input  n=3}
# This function has been saved in the gluonbook package for future use.
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root)  # Expand the user path '~'.
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_workers)
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False,
        num_workers=num_workers)
    return train_iter, test_iter

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
```

## Training

Now, we can start training AlexNet. Compared with LeNet in the previous section, the main change here is the use of a smaller learning rate.

```{.python .input  n=5}
lr, num_epochs, ctx = 0.01, 5, gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
gb.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
```

## Summary

* AlexNet has a similar structure to that of LeNet, but uses more convolutional layers and a larger parameter space to fit the large-scale data set ImageNet. AlexNet sits at the boundary between shallow neural networks and deep neural networks.

* Although it seems that there are only a few more lines in AlexNet's implementation than in LeNet, it took the academic community many years to embrace this conceptual change and take advantage of its excellent experimental results.

## exercise

* Try increasing the epochs. Compared with LeNet, how are the results different? Why? 
* AlexNet may be too complex for the Fashion-MNIST data set. Try to simplify the model to make the training faster, while ensuring that the accuracy does not drop significantly.
* Modify the batch size, and observe the changes in accuracy and GPU memory.


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/1258)

![](../img/qr_alexnet.svg)

## References

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
