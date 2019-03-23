# Naive Bayes Classification

Before we worry about complex optimization algorithms or GPUs, we can already deploy your first classifier, relying only on simple statistical estimators and our understanding of conditional independence. Learning is all about making assumptions. If we want to classify a new data point that we've never seen before we have to make some assumptions about which data points are *similar* to each other. 

One popular (and remarkably simple) algorithm is the Naive Bayes Classifier. Note that one natural way to express the classification task is via the probabilistic question: *what is the most likely label given the features?*. Formally, we wish to output the prediction $\hat{y}$ given by the expression:

$$\hat{y} = \text{argmax}_y \> p(y | \mathbf{x})$$

Unfortunately, this requires that we estimate $p(y | \mathbf{x})$ for every value of $\mathbf{x} = x_1, ..., x_d$. Imagine that each feature could take one of $2$ values. For example, the feature $x_1 = 1$ might signify that the word apple appears in a given document and $x_1 = 1$ would signify that it does not. If we had $30$ such binary features, that would mean that we need to be prepared to classify any of $2^{30}$ (over 1 billion!) possible values of the input vector $\mathbf{x}$. 

Moreover, where is the learning? If we need to see every single possible example in order to predict the corresponding label then we're not really learning a pattern but just memorizing the dataset. Fortunately, by making some assumptions about conditional independence, we can introduce some inductive bias and build a model capable of generalizing from a comparatively modest selection of training examples. 

To begin, let's use Bayes Theorem, to express the classifier as 

$$\hat{y} = \text{argmax}_y \> \frac{p( \mathbf{x} | y) p(y)}{p(\mathbf{x})}$$

Note that the denominator is the normalizing term $p(\mathbf{x})$ which does not depend on the value of the label $y$. As a result, we only need to worry about comparing the numerator across different values of $y$. Even if calculating the demoninator turned out to be intractable, we could get away with ignoring it, so long as we could evaluate the numerator. Fortunately, however, even if we wanted to recover the normalizing constant, we could, since we know that $\sum_y p(y | \mathbf{x}) = 1$, hence we can always recover the normalization term.
Now, using the chain rule of probability, we can express the term $p( \mathbf{x} | y)$ as

$$p(x_1 |y) \cdot p(x_2 | x_1, y) \cdot ... \cdot p( x_d | x_1, ..., x_{d-1} y)$$

By itself, this expression doesn't get us any further. We still must estimate roughly $2^d$ parameters. However, if we assume that ***the features are conditionally indpendent of each other, given the label***, then suddenly we're in much better shape, as this term simplifies to $\prod_i p(x_i | y)$, giving us the predictor

$$ \hat{y} = \text{argmax}_y \> = \prod_i p(x_i | y) p(y)$$

Estimating each term in $\prod_i p(x_i | y)$ amounts to estimating just one parameter. So our assumption of conditional independence has taken the complexity of our model (in terms of the number of parameters) from an exponential dependence on the number of features to a linear dependence. Moreover, we can now make predictions for examples that we've never seen before, because we just need to estimate the terms $p(x_i | y)$, which can be estimated based on a number of different documents.

Let's take a closer look at the key assumption that the attributes are all independent of each other, given the labels, i.e., $p(\mathbf{x} | y) = \prod_i p(x_i | y)$. Consider classifying emails into spam and ham. It's fair to say that the occurrence of the words `Nigeria`, `prince`, `money`, `rich` are all likely indicators that the e-mail might be spam, whereas `theorem`, `network`, `Bayes` or `statistics` are good indicators that the exchange is less likely to be part of an orchestrated attempt to wheedle out your bank account numbers. Thus, we could model the probability of occurrence for each of these words, given the respective class and then use it to score the likelihood of a text. In fact, for a long time this *is* preciely how many so-called [Bayesian spam filters](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering) worked.


## Optical Character Recognition

Since images are much easier to deal with, we will illustrate the workings of a Naive Bayes classifier for distinguishing digits on the MNIST dataset. The problem is that we don't actually know $p(y)$ and $p(x_i | y)$. So we need to *estimate* it given some training data first. This is what is called *training* the model. Estimating $p(y)$ is not too hard. Since we are only dealing with 10 classes, this is pretty easy - simply count the number of occurrences $n_y$ for each of the digits and divide it by the total amount of data $n$. For instance, if digit 8 occurs $n_8 = 5,800$ times and we have a total of $n = 60,000$ images, the probability estimate is $p(y=8) = 0.0967$.

Now on to slightly more difficult things—$p(x_i | y)$. Since we picked black and white images, $p(x_i | y)$ denotes the probability that pixel $i$ is switched on for class $y$. Just like before we can go and count the number of times $n_{iy}$ such that an event occurs and divide it by the total number of occurrences of y, i.e. $n_y$. But there's something slightly troubling: certain pixels may never be black (e.g. for very well cropped images the corner pixels might always be white). A convenient way for statisticians to deal with this problem is to add pseudo counts to all occurrences. Hence, rather than $n_{iy}$ we use $n_{iy}+1$ and instead of $n_y$ we use $n_{y} + 1$. This is also called [Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing).

```{.python .input  n=1}
%matplotlib inline
from matplotlib import pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
import mxnet as mx
from mxnet import nd
import numpy as np

# We go over one observation at a time (speed doesn't matter here)
def transform(data, label):
    return (nd.floor(data/128)).astype(np.float32), label.astype(np.float32)
mnist_train = mx.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test  = mx.gluon.data.vision.MNIST(train=False, transform=transform)

# Initialize the counters
xcount = nd.ones((784,10))
ycount = nd.ones((10))

for data, label in mnist_train:
    y = int(label)
    ycount[y] += 1
    xcount[:,y] += data.reshape((784))

# using broadcast again for division
py = ycount / ycount.sum()
px = (xcount / ycount.reshape(1,10))
```

```{.python .input  n=9}
for data, label in mnist_train:
    y = int(label)
    ycount[y] += 1
    xcount[:,y] += data.reshape((784))
```

Now that we computed per-pixel counts of occurrence for all pixels, it's time to see how our model behaves. Time to plot it. This is where it is so much more convenient to work with images. Visualizing 28x28x10 probabilities (for each pixel for each class) would typically be an exercise in futility. However, by plotting them as images we get a quick overview. The astute reader probably noticed by now that these are some mean looking digits ...

```{.python .input  n=2}
import matplotlib.pyplot as plt
fig, figarr = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[i].axes.get_xaxis().set_visible(False)
    figarr[i].axes.get_yaxis().set_visible(False)

plt.show()
print('Class probabilities', py)
```

```{.json .output n=2}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"61.688136pt\" version=\"1.1\" viewBox=\"0 0 572.4 61.688136\" width=\"572.4pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 61.688136 \nL 572.4 61.688136 \nL 572.4 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 7.2 54.488136 \nL 54.488136 54.488136 \nL 54.488136 7.2 \nL 7.2 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g clip-path=\"url(#pcd8e0a022e)\">\n    <image height=\"48\" id=\"image0ec254979d\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"7.2\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABHNCSVQICAgIfAhkiAAABMhJREFUaIHtmW9o1WUUxz/70zTLaY7WmGnbLjKZLGyLHNtijZQmaOUQltAfVkGZaEZJVC+jXgQRSBTaP19EUU6i1KIyY0jiRo6WbG6YNZ1JrlyFOTRd9eJ8z+/5MW63K70Yd9t58zm/Z8+993eec85zzvMsawb8TQZL9ni/wP+VKQPGWzLegNx0J14mThcLgGull4hFsbnnpP8kDgKnpJ8Wz4h/AaPpvsgYmRwemAmUSb9NfBCYu0oPa8RGMR/4Rvpn4ptw4AdTP9HQPvEY8Jt091y6kvEeyEpVyDzeq4C10ptvkPIKUOOjG8SF4kXghPTd4vOw/aSpLxgOfG1sA/Zr1jHxDOnlRUoD5omtwJOlenhHrFkfe/ErxT6xX0ZAWAaALwxH3zVuNpzdDFs0w8Orl5DkqQzJ+BBKmcTF4lKwrAWoqZCSwOIIuPiS0ZfvCHC19EpxcSHQoI+uND6zE4ArZsFjz9pQjqaPYl6A4IlkMrE9UCBWTwPqfbRa7IAexfImw4A8MEBwwKI5UlYPQet202vmGwtvMrZ2kjUi9UXjMGFr/V68cKkGePWlmFBmo6T8CnaZdlwv/rL+0h/7bPmwccVWqOvS4OPHjS1iaTGsth0q/4gNrfrIIhFCBXeD4jKxQyhyWQ4hu6Ja+Ue0JO5iZ1/ss76K3wEntO+3PB37XrDVX+K6YXE33KiicEh/mnwe8K6RQUJbmXAPlECxxbAXvHjJUk5G/jpHKEiz1BM1bdVAMVCXZ3r9n8ZGaNhmqu/O7uG4TGwPeMwNnIcSryp13q0koFaaDgblan/6CMXHc+F3Qj7sEaukFDYCVVp5d2ctVG4z9RoNxYtcWgZoB6QDKGnXQ2unPlkO1drkm21mg3qb7pgB8UOLL4gb0i0u6yKEaGmhsWyIPC3MbO8Lk8jEDiFPwC6g5VM9fCku6yfa++61NLtFc/r7w8oPimeBvDHf+7P/0KnYoHe2BUN2ksK8928yOTzQCxz8xfTqDzW4pBPydaasVt+60dqBhzbAqLLXHXcy9r35YrTtTie4xz2QTbQD+HskOxekNMBdN0jYOarfk1IP3HXQrTHc90H0to+oKZupgtBBSOKE6I05FcB1Y15phCj+olBLIhM7hNxlpwlb3j6F0s1vA5XqJhdpv7tcB5UndkYJeM9bxtrDYfucKy708/VyIPd6PegHeuHoeVOT9UAuE9sDLiOEPmSvWLEbCjyIn9KdwlXLjfkrYZMO8JWWBIl2SPRrvp9VNZ2m+UT19tdvje2g5nXKA1wAfpTeIc4D7n8j9gCwVn1j7hrgAdOb1Ac0dRPFN7PFW8UioisXbz3brIBCag+kfbnr7bG3cnuAOWqW7tSlRHSOfHgHsE4PLeI6YJp0N8ANeh961GO9bvh4JBxk/LeTScaHUMqbubh4KztDLMKuHAFWiM2enBuBR6XnrZdyB+Gew4/pOww9r8Fzph7SRcerwOdjZierxJPHAy7xf3T43c8CcanYDBTdroe7xXrC1YzvybqWYQvsPWxqm4b2EzrZVFful2yASw6hGfOw8jusBYTw8pvFMqLiHPU2vTF6pfdqPYy14P8lky+EUoknenZMj4+5eJc7OoaQ/PowlUx5YLwl4z0wZcB4S8Yb8A8VoSxPiqjWPwAAAABJRU5ErkJggg==\" y=\"-6.488136\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 7.2 54.488136 \nL 7.2 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 54.488136 54.488136 \nL 54.488136 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 7.2 54.488136 \nL 54.488136 54.488136 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 7.2 7.2 \nL 54.488136 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n  <g id=\"axes_2\">\n   <g id=\"patch_7\">\n    <path d=\"M 63.945763 54.488136 \nL 111.233898 54.488136 \nL 111.233898 7.2 \nL 63.945763 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g clip-path=\"url(#p767374a98d)\">\n    <image height=\"48\" id=\"image2b8baed964\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"63.945763\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABHNCSVQICAgIfAhkiAAAAwNJREFUaIHtmcFqE1EUhr9aU2JraCuIYimVglYRVyrtQuhSRdx0oW/hI7h05wv4AG5cCYKISujGqthiESQqSKUlCLWxGluCIdHF/U/uTUxxkiaUOPPD4T8zc2cy95z/3Ln3pm8QftPD2LfXL7BbJB34F/pl3ULPZ2B/1IYWxbR4EBgJ/PBaFUjJtwgVgC/yi+JyK2+6A+KRgRQwJP+I+CwwHfjhtYHg3pJ4GXgs/7l4XVwGKtHe9y/EIwPge2rRHQHG5Z8XH5gILho2HFXW4K1OHRJbLYTRbzUTkTpQxRecSWIz8GvDpHXgOL6K3znKrPl2jcXbr99oB/GQUIX6yAOsAp/kWzGOWYin8DrZFi/AD7n2rEoDg89SVCnFIwPgNWqFl8cXpfGYpecYcEF+wVE6uNeS0uxD1pUiDh9sXADey38pvmwn0sCZk84vfgDcwGQdaJRQleQ70DpKOBkBvBYvShvnCgBX3cHMQQAGJpYofXanTDomy3ajD/9BBtruQAXYkuVkD2VusjMmu+3siiuNtO4NbTeIbwbAabiKG5EKQFbGA3BVUgJmnd2EozjrJDqSgbJsRXb/J8AzmYRz+pp1pSalTiDeEmqEiSYLsJh1FkhpbhjmhmESZyn8pLVd9HwG2v6QhbAo2ExyHeCODu7NyxmBG867eNdxPmjf7nDakQ40ogLwxI5uia/Xlm6X1IElXdnCz5NaRSIh8NIJpfTtq/NHX7xyzsxQ7SMwq3aPxHn8DLXVvaIkA+ALMMyELTNHn8oZz8Iv5/ZNOZ7W+iGHrwFbE0XNRJIB8JEPuTaqvBHP49OiBf8JHZ7CD6kW+SLRhtaODqOpgG3da/tCHA4alutPTQIf5Ycbv41Lz2ZIJBQi3NOxYkTLSJaDhquOLHoZ/DRblyhSq/kkAzvCImPat4ht4yOZUzFMLvj7rJZXxJs0H4qjoCMSCkcO8AUJXkGZoJ119Lt4g/qJHbggRNnw7XkJ9XXrj+5W/pmM9b5QV9YDsPv9nqjo+QwkHdhr9HwH/gB9jcLx/ogBngAAAABJRU5ErkJggg==\" y=\"-6.488136\"/>\n   </g>\n   <g id=\"patch_8\">\n    <path d=\"M 63.945763 54.488136 \nL 63.945763 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_9\">\n    <path d=\"M 111.233898 54.488136 \nL 111.233898 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_10\">\n    <path d=\"M 63.945763 54.488136 \nL 111.233898 54.488136 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_11\">\n    <path d=\"M 63.945763 7.2 \nL 111.233898 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n  <g id=\"axes_3\">\n   <g id=\"patch_12\">\n    <path d=\"M 120.691525 54.488136 \nL 167.979661 54.488136 \nL 167.979661 7.2 \nL 120.691525 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g clip-path=\"url(#p529ff754ae)\">\n    <image height=\"48\" id=\"image6bfc8eb7dc\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"120.691525\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABHNCSVQICAgIfAhkiAAABJhJREFUaIHtmdtvVUUUxn/Qnqb0tEWqyAlYrQW0AQyGWk0ELBoIIUSNhBd89M/wyX+DBx/VeElM9IEghARQtDES8NJYvNVUTGmTWos9HltafZjvOzM93W2O8QHPhvXyrZk9+zLzrbVmrdlr2uBvGljW3u4P+K9ydwK3Wxp+As31DmwSesYtrDz7poy+hYy+xYxrqT5fx3flm4FW4UagW/p24TbgXulehZvCSWBK+rSwnDx3ruY988TVvpmg751NxtVKPhkoCh8Q7gdekn7AnceBPdJt9KPCS8CnQb0uKiaJ9u3hBWEZGJc+Ivwu0f1Ys5r6SeYEOoSPCV8EDjyhxmvCo08Cr6hhY/tGeBI+mQFgy/nQs2WEaAse7plUgOGgDn4d8FzyHQvJMIhmBnk1IS+QHXcXwLNq7PeoFwjcAJSEXqPDsPftoBqvzUQbsBe3CCepMtB5LuChj+JK27zGhGUiK/lkwOJZFiGulpel8zzQrsYu4UMJ2j92BnhkCPgi6LeuBfxVQypEg1dsLnZBz9SSruoQE7niBGZrBo4Bfd+qccVPOwOdN9Q4vPRj2UmMYc01CDT/EnBau8NXwJCufRZgYiruA4UaTCWfJuRd80fhRaDvtBo9wg5g4Mugr7MpbUqecp/Q1ypUw+yfesNlXToNnA3qlb8CjgAm3U5sWUvendjBcFT4MVD6I+jPn1TnNNGhBy8FbF+vjm1An/R7hCWqDHkDsxMPwUWt/Ofq+pm48h6WbmCWfDJg+3IUupz0jYuJQ69Dj9KEahA6ckoXT0HLCXUOCtuB34Paophc0I5Wif52VTjG8kzWDLiOAFhTz6lEEeiS3pugcyXndAPmc18yKWP//USzUsLzoZboVXhXXW9pRJo6OY3OSqvzaUK1Mkek0ynuONX0BaUv9IjbHRdg8IL6HH5PTMDLE0HvFJ+mczd0i4G25L0OJquVlvlmoKkGISaSZeLKmB1ni8NEdo6LiYFWYKs6D8r2ne7ugd1vLO0aJjqr35N1MFCXCbUSEymn2m0sr0vSujbdxQEGzhID0lPJQwB6Yd3DmsBPATuJ5pH14ZZ8m5Bn1wZslu7UtovIimtoh7kKkX7j+CKUbGOTwq16Q9diOPoANouB9WSfL630jQ0rqzLgGqaVWDTuEPaynBWvxiQxBJqlUiF5oCPBnPiZJxg9MXMqUt/q5psBS4HlqcRzQMnpv4/r0nMQJy7F5MZ90k2npUyVFftMVvWVJXXVxOnRn0NnqQA8rcZBoUtj21R6Qxew4UE1VOTMabeYpLqJzGhEmaVJ20qSbxNKT8RcXDgC/jYPG8yzvfkZr8dRoF96j7AduCVdCdKMGLgKswqfaa5193g9TQ28B7nk2wgce18Ne3i3rLb/B2CvOh8VVohnJ2LgHTXfhDNSXdCkoXg1qaugKZA4r3A7MaWxDz+ufIZjxLzHZ10Von28F2D2g9isncAN4s6+mjS8CdXFACw/0++gmr5UWXEqvIno19pgmSPW2KPC74XXCSsOq/+NyZI7h4Esqf1zmRZAtX0LxI3JqVDWX8p/K3c2A/8HaXgGGn4C/wCOlB3qFnQYOAAAAABJRU5ErkJggg==\" y=\"-6.488136\"/>\n   </g>\n   <g id=\"patch_13\">\n    <path d=\"M 120.691525 54.488136 \nL 120.691525 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_14\">\n    <path d=\"M 167.979661 54.488136 \nL 167.979661 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_15\">\n    <path d=\"M 120.691525 54.488136 \nL 167.979661 54.488136 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_16\">\n    <path d=\"M 120.691525 7.2 \nL 167.979661 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n  <g id=\"axes_4\">\n   <g id=\"patch_17\">\n    <path d=\"M 177.437288 54.488136 \nL 224.725424 54.488136 \nL 224.725424 7.2 \nL 177.437288 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g clip-path=\"url(#pc7affd4af1)\">\n    <image height=\"48\" id=\"image027c328da1\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"177.437288\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABHNCSVQICAgIfAhkiAAABI1JREFUaIHtmUtslFUUx3/D0NFOrfYBtOGhlfCohgISI40LCTsFgZiALlw0Ji7AEBNdsCAxISYa98aFMSYSVKKJIVET42thYhrEF0iCFIjxgQ1YaaaWkpZxGBfnf+be+dpMx7rAmeFs/v97vzt3vnvP455zv1QWitSwzLveL/BfpeYXML/agU3CtPDmqC/5LA0UxPPCyYhfFV4T+ti5SH1qIN5lgFagS3y5sA/YIO7YeYtIBoqjRk+paxD4Xvy08DfhGKYh+PfaqHkNpGYKo77z3cLlwCbxR4W3bweeVGPbrSKPCZcAvxstvmZ4EHjF6LffGL6v0YPAWfGc0P1lNqkvDbjtdwhXC/uBx8VXPSTyNPDgYjW2CjcKlwIt4lPC48DzRl/+y/CAwaHRoI3jwhGq00LZAjwcLhHeI9wC7BTP+AI2E2zMV+6emAV6xde7eW0FFoi/Z/DGsOFzcOi80SMa8R2gOFDRsWvehMrCqG+ka8KNoEBwst6PNHYQGDf+q06kkWguD7ft98lcnjgMA+psvttwhzQwBDtfMqoehsP0DaSBQgJ9R88RdkMRkMkxuCQu8y2FwAJBe/ceMxw4Bi2/qHOPjjePFhugWRFj9ZDhMkqBuORasy7AcxN/2QvCPHYaQ1DZVPTCE4k/ukY4S3yuFmDgTTVkQXhAWEjpqO/UAm6L5qgk9WlCV4RuQuNMd3BvQ9iFlqivKfEsBxRlaymfeCoanCkfn6c8u43fL5b60oBL0mmuEHbU7bKDECq7E9gUje8UrgFSa9RYJsxGfyiHuqiu3AzvMZPUpwbc1uIc3fvczrsJGaqnGe3rRJYy3YC7CKnS/UJPds4CJ416/TBC8MVKB1nFkjL+YTrxrDt6j3bPsZ8R9i8GetT4O8LL4n8aHFW2cwQ+lUP7Ai4SSs9KUp8mVElcKxOEg6yknhXesZ+QgLcJTwOfGy3uNfzQ4MInJVoqN3NUl07XtwZ8Y+dRvvMAPwGfifceNuz0uHpgL8z3qT0FXQGcKJ/ka4NThCx0TFhtSVnxVsLVk2H6CTwKfCnu0WrHC4b9HwP7dltj14t6+lQY6aGsx6CLkNfFJ3g1NxQ1b0Iz3kr4Lig9IUvIRlujZzeJ+4Z6hdwHPCK+yH15H7B2rRrKc9/+w/BZOKgj+B2NOEEjFjRJ8d1eSNhdx7boeRPlMond9QBsecsw0woM/GANv8q7Q7gJNr5rVH7NMMHXG08DybvRDkIRtS7CleKpBYkfThAcwy+XegihxnOE6ArQleEazkbTVQqpFZM5xzThPO0TrroLeFgNT9J6hLF9LfJQsJ7wml4cq2C+GkJxPoGzSX2akBf37kSXgJ/FHXvHCGbiTnnnAyLbCWrxMgfgK+F+AxXwfBGu3l034zTyQebi4bGN8kMKrJjxw6rZ6wHHbUBmsxr+1WMIfjxj9HV1vWrwwWXQhR9HhecJFjDnBbikKV8M2Nngpa2Xur64lYSA45EkRzA/pXSl68pzhEssv9DNc8OEZpfkXVGMyRIUQmhMfsEsMPcvlo2tgf+D1LwGbizgekvNL+Affq0EcHzmrmYAAAAASUVORK5CYII=\" y=\"-6.488136\"/>\n   </g>\n   <g id=\"patch_18\">\n    <path d=\"M 177.437288 54.488136 \nL 177.437288 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_19\">\n    <path d=\"M 224.725424 54.488136 \nL 224.725424 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_20\">\n    <path d=\"M 177.437288 54.488136 \nL 224.725424 54.488136 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_21\">\n    <path d=\"M 177.437288 7.2 \nL 224.725424 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n  <g id=\"axes_5\">\n   <g id=\"patch_22\">\n    <path d=\"M 234.183051 54.488136 \nL 281.471186 54.488136 \nL 281.471186 7.2 \nL 234.183051 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g clip-path=\"url(#pec8c9b862b)\">\n    <image height=\"48\" id=\"image42cd468f47\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"234.183051\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABHNCSVQICAgIfAhkiAAABEJJREFUaIHtmUGIW1UUhr+ZmDEmVuMMbaXatArTysgUUcGCLqdYUNBF60ZXduvGvejetUvRnTstiBuhdnBRxgoWC6UglGrHsXQYlRnClDQxqYv7/+/ePJ8xU4WSxLP5z7vvvpd7zn/Ouee+TFXhNiMs03d7Af9WRt6Ae+70wRJQkX5vMpa/VxZ2gZ70bWFH2Er07g7XMXkM2KM1YFb6w8L9wt1AVfouYYno+d+F1xPckL4p3CayMkgmjwHH+TQxzvcKFxI0O3VhFWhKXxd+L7xM9GRb2CHmzKC8mDwG7I1eToeYH7uBA4kOMFWF9s3+eT8La/RXMAietXcHMbBjA7zYDqD1ZAno5GwDc9KnHpVShxlNnFMMOQTTRead8k8yGSFUIlpaSsZbQifnL8J1oOQ4cW3dR5bZc6qV1VvxXb0c5vW/k/FmwN6uENuFWnLPY3Z2HyMOZjOwSNyltgLMXozP5z05bEsx0AAvbJZYVQ4K68QkrCZjAI9DCBkICwdYSl6sLXZRBqwQd+y0h8ondpGMdwiZgQPAMekvChfLyNXAg0K7rw48Lf014aFjwP1B33sagAeU9UufxlLsgtDmr4ldxMR4MpAmL4S4f1n6Ey9JeQt4VrqDP617M86MV4SngCeD+tgzAd99B4CnrsCG8sEda5EU3RtPBmxVOUFXwywZjh8BTMea8DvhleRtN4S/Ek8Orwc48lHApatZRbqgGevE1sR54chIc6HQgHzybAK/Sa9lT1eSx78K8ImOKGeBphrjw8sB316GhzxfoeSkLsc22nvJNsPtBeMZQrbc3tggUtv4Wsob38Ie7cur8vyHAc4tww+a5v3s+OfAyolwcd8HGr2RgVvr68kdb9yDDvzjyUA+WbaAS9Jf/ULKZ8Cbim+7SnIV8DR7r3kRTr6vi/eUvPwRoJ49mpXKm8TzxqDD/cCd2EncAn6S/o0Gj54GHtGgd+LDAZ5bhjMaciidB05+rIsTF/rmcxDmpfoEV3TYKZLxDKF8F9gi9ioq1xw9Q+yF/DliJsChWXhBRdzRtQmsXgt640sNOjbKsEcszm/F39nITZucJM5Lh+wMkuXCuR48b096y7SLdsGCxrwnbxOSG6BxNs4Dgqt13DQD80QGXM4nl4Eusbx5o7kE7P8x6A0zII/eXssezQoURKJYEbrk3CIrea6o+5Lbgza0gQZ4YofYULk1u0wsdQuivSpsAsrXrJa3iSGxKksarrUVaF+L8yCcvW2AneY1pAaMdwilG5kttTc6xLBycvoIA9Hbxi5ZleW8cGstPmdmzVyTCfkuNDXMv5Ql+g83EGLUVdCen0nm5EtfjxjT7lD9Cb5GTFCfO9aJrJjhohwYyoBU0k+L+c+N6b1ubmyaaLzHfF1JdH9tbBFD1M4oauomI4T+CxnETpEM+2lx5Bm44/+JdypFHt3pf8JFMvIM/G/A3ZaRN+BPGuYS3qiC+qEAAAAASUVORK5CYII=\" y=\"-6.488136\"/>\n   </g>\n   <g id=\"patch_23\">\n    <path d=\"M 234.183051 54.488136 \nL 234.183051 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_24\">\n    <path d=\"M 281.471186 54.488136 \nL 281.471186 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_25\">\n    <path d=\"M 234.183051 54.488136 \nL 281.471186 54.488136 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_26\">\n    <path d=\"M 234.183051 7.2 \nL 281.471186 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n  <g id=\"axes_6\">\n   <g id=\"patch_27\">\n    <path d=\"M 290.928814 54.488136 \nL 338.216949 54.488136 \nL 338.216949 7.2 \nL 290.928814 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g clip-path=\"url(#p9efdcdbc94)\">\n    <image height=\"48\" id=\"image19912e3f98\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"290.928814\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABHNCSVQICAgIfAhkiAAABIRJREFUaIHtmE1oXFUcxX/JmBJSpySRftDopDQUa1AT0hQCYzeCiKKuih9LwaVrd266EjeuRBB0WcGFIEpFqYpoU6tF0hoL0lgnqQlJ06ZJU804Y6KLe07uy8z4miCxzKT/zTn33jdv7v1/39fUBn9Tx9J8uzfwX2VTDpDZjJf+i9y10R+0CFuB7eJtiTkIByiJ/yn8HSiKrwhLifHyRjciqXsXWpcFMkQt3ye8HzgiPlCBmd2sqvvaQsCvgW+0flE4LrwG/CFeFq7XInVvgaa0NOpgzBI0DvC48CXgnmc1eLFisWkA+CvwpfMBTwDvBDrxScD39PgwcEl8VlgkWiNNGtsCzirdwNPirwi3vwq8rMEurx4SthPD66bwFFz/KNA3NPVmgONzcFJTI8IZYFE8LR5Sg9gH6AGe8sZfEHk08cCSNlYUlolRf/cOkX3QsSvQo1fiLoHn3g5pFmKqLRNdyGu1pO5dqKYFHLwuWvuBPi/anl8BH4s7L04Jy8C94kM3Ag6ehwOas+X00kwf9J8LPBnMdiFbpZYrNaYFfFIrqhm4LN5zQs+8Dxc0ZwNYY23AAWm0/7QmnwEeE98pbInjbtE9wmzi//1YLQukBrGDaJzgMQBnlVSmiYeaE9rUWaI3tWrx4JnExvsr/iBbu6/yxtOkMV3I4vRVIFrAZq1VJbMJrNLeAtFULrctFZiQMrFrTWvPG9sC9umpxJyDrBt4SNwdarswC+wW79gr0kt0dEej/2Ae5kVtpBLxvpBWiRvbAuUE2h9tgaNA/gENnhR2CotE7bYm0Bawz08Lp0KciQLBEpV3hA0fwMHTTAxQV+R8N/CuBkNPiOwT/gScDXRS2ygQC4bLrQrJjVEY1ZRT8+wtNm5pbBfy6TJECzgmGQCG9mtwXOgw/hB4XXw4wBQwpqkzwi8CfJqYKggXuRPEa8VaWPXLEsBvGhQrnh6FK9L8B5o6DXwb6MSvAX2JGSb2VZVtya0k9QDOPMvEjODEwShwSkbOH9NkXvgZ/CCq+y8nYUSn99KPwhmiuzhp+X8hKq1YMYat4kJlYl+kWyDXx6HDF5retwJ22KXG1nSaxk75hyu3pZ1waYLYks8TNW63mk6MbZ3GtoBPuUJ1DIwBh5UGV6vb85/H13rOptsDORWyXEFz6kpLV6vvFrPElOr6p9LIIlvFApZlqv1xBBj8LvAmV6Fe2enhTsip5OWuBjxSgsmKl0jt2y5Dj3jPz1q7AJP6rupvqu6TLib2s+ED+C5yidUaS/5LEX9/bJ+DXE6DwQBdQJffokMtTQQsALpDr3aLGeiSYrIpTdHWcKEVqi1wjkSnLO0dek0TY8Aj0m6fcCexvbVGnTNnqHIrJuEXPecqbRdygYUGsEDqx92kWHnWeiexMz0odOZ8EDgsvs1f6PYSLzR+SbI6Kj9PK3BHiGnze6FbkAViGl33ASoPktxHtgaa+5K2g/jNx+9w/7NI9CZ71xzxnjybeA7Wttdbx4U2IkkrNdeYS/vOU9W20+AXmk2xwP8pdW+BOwe43VL3B/gH8so33nUz9vEAAAAASUVORK5CYII=\" y=\"-6.488136\"/>\n   </g>\n   <g id=\"patch_28\">\n    <path d=\"M 290.928814 54.488136 \nL 290.928814 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_29\">\n    <path d=\"M 338.216949 54.488136 \nL 338.216949 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_30\">\n    <path d=\"M 290.928814 54.488136 \nL 338.216949 54.488136 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_31\">\n    <path d=\"M 290.928814 7.2 \nL 338.216949 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n  <g id=\"axes_7\">\n   <g id=\"patch_32\">\n    <path d=\"M 347.674576 54.488136 \nL 394.962712 54.488136 \nL 394.962712 7.2 \nL 347.674576 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g clip-path=\"url(#p728b23a716)\">\n    <image height=\"48\" id=\"image25bcd1567b\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"347.674576\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABHNCSVQICAgIfAhkiAAABH9JREFUaIHtmU1oXFUUx3+ZJJM0H9ZUBkNrtMngaGK1IJFoUaoFEaSIi266ki4EXelG1+5cuNFNRQVXVnClIrgRBb+pSqBtaBzbRGrakBqNNWmLmWRmXJz/mfsyeYwvziIMr3fzP/e++z7u+Z+ve19bD1Rp4ZbZ7g9ott1YwHa3GwvY7tbyC+ho5uZ2Yaau3w101o2VgDXJ9Vhu4hvSx4Brth8YlLxfeED4MDAk2bVcBL6UPCn8WXgZ+KduflJW2pJm4m5hTjgBHJP86FMSXhcOPws8oc6C8G346LSJrxp89oPhh8CPmjUvvEZYTKPW8iaUiIF2Nmoe4HngkefUeXNEwsfCfQSjuCq8ApyS/LLB8VnDF+ENqftTzSjqDmjMRDoY6AVcx0eFL4wCn6iTf0/CPcJvgfclTwkHgYMRGWqMvXsaXjLxraVw5axmaSjWsRNFoU6CCXnE4QCQ36XOH8LjBlffgW805FazsAx3/mLyYY3dXjA8Qi0kHXnNcI5gQm6M12MW0vImlIiBDBb3AXb74F6APnU+N1iXTZ0EVuoeMkXw8S+Ez4iR/cADJt4yajg+Dec0bVHoTKSPAQgrrfhAGUKIvGDg/nqOkIofE+YIMfJv4cnIC1ytui8/HVzd2XdnjobV9DDgq/bIwAKwLp10yBf+1LUisCr53giOS3amloXXCD6j8rWLoPnsxktbX0CFEMLmhA+eBZRIKci9etW/APwk2U3jKHBfj8lDepp76Ty2aMIzl7B1QTCTuDyQDhMqExj2apEp4DvJI78b3qp+N5SUyLJeO08Ch6R5j8Wu4iK1xFcWE7MEp/V3p5eBCiGJuA/MLEHey4X7ha7ZfUFbH0jpuRPw0AmTdwzrovvMIvx12cTvNTQZeZczEFeVJjYhZ9uz4hkg7ybk0eWQ8C7YoYxamtZCCGlg6FdDr6QqhG2PMgoz2E4tuoC4lg4TKhNMyB1rFliQdge/0uAeYQ543MQJzZkk7IU9wnYJs9ipBYQkfYX4zFvf0sFANJF5Jp4jaPTJryXcIZzAdpVAQRuIg6dCCPaNij8zQ6ixnInViNzohCIdDECww2hC80pg5qJhLSrdTNjCqRp9+gzMVzY+wyuRFTafB5VJdqzS3gmv/NekKlZIVbEVrwm7MZPqwkLe6BzwG7bPGcBsZADYaZOGijCqDx7WvQOYuZT1zApmFmWgTXKjTXvLm1AiBsC04UysC7OYNjNYgstW4WIVblvGauFF4CaMsn7oW4e+PbD7Eoxhmh/BiFoFejDH7SCYUhuRTVRMa3kGtny46461SigrvGbxBLVrHgru0H4anKNWM+1VtjrsewA2b1ai9Zcz8L/PhaIPcVzGKIdQx/gBcD/Qe95kT86Mseloo6AwNLYWaiBXykpkAY1a+kyoFJE9K8ftVV1742Li7vPQK8qqSsGeB6L/B0oRTPKPoOUZSPyDw1v0v5ifFrgv+P6kn1DrR8923J9d29Gs7n7krF4nJT6wZQbiWpwPeNt0ohfTmvlL2dRv1iQf0MzHJWktb0L/AqVINvA4yzEhAAAAAElFTkSuQmCC\" y=\"-6.488136\"/>\n   </g>\n   <g id=\"patch_33\">\n    <path d=\"M 347.674576 54.488136 \nL 347.674576 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_34\">\n    <path d=\"M 394.962712 54.488136 \nL 394.962712 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_35\">\n    <path d=\"M 347.674576 54.488136 \nL 394.962712 54.488136 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_36\">\n    <path d=\"M 347.674576 7.2 \nL 394.962712 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n  <g id=\"axes_8\">\n   <g id=\"patch_37\">\n    <path d=\"M 404.420339 54.488136 \nL 451.708475 54.488136 \nL 451.708475 7.2 \nL 404.420339 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g clip-path=\"url(#p9117ebcde8)\">\n    <image height=\"48\" id=\"image1c6d920f49\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"404.420339\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABHNCSVQICAgIfAhkiAAABKJJREFUaIHtmc9PXFUUxz/MAIPYaQmoJSBCChYLrZ1QG1EqLixWkyYm1fQ/0JULN+6MvxZuXRm3xo1LozEGq2nV2KiU1KJNY0MgENJGo7SFkca0Drg43zP3Mo7Thy7oTDjJ5Pu9913y3nnn96OuGdbYgKQjnirZc2woc2YVKIjfLMEC/11Stz5ye0t9kkNpoFm8RdgNdIn3CO8WNgF/il8TzgML4r+VYD46v1FrJFKgAbhTvFs4BDwm/rCwbadIM7BkdPGK4RngW13+SegPfUO/WJIqUtsulC6ztxpxd6u2VpGHhJ2YXwBtZw0HLsJlXXZX8mAvd5+kUtsWcD9MASvii8IFwptck5/XeaD0E3Kk5L4FaL9uPKu9ON26FWILJ5HatoDLKiFjeOqbB6bE9wlz7uRNmBWcA8xA14RRt0Djhh/3n5JIgTilyQtYiJ6tQ5j7XuQI8Lh4p3AO+qVAu7Y8CWwFcRIpLSx5YE7cC9S4Avep74BntZlTjl2+QuM3RvsmDb2qNxGsUBL7t5Sqt0C6AV5PdJD12qaxNvYmZp1r4jPA2DSkHgV+AfYNA+3QnYKlqzAI93xufdQUsAP4FbNoGvgLqNMvSZucSAF/+DrM59KY2Zu0rscyVQOwrLODdwDTwLFV4CrwIhw8AcOQmYBML+Rn4F4dywMZKVCvl5KkJlS9C9VVGmjiAcWrZlO0ly3ZaxPuBd4Qb/Ri8eB7FJvr8ZcAKDxty9eAr3TMS0me0KFWCuyqt0DFGKjH/D6Dve0MNhdkgG1YIWrU2mNjDfPdFeA8kMsDp4FjWeB5YDf0TUFfN6n5eVI5uHHOgj9LiIUC9uZTWFz8m1S9BcrGQOlw3gK0RhxMc88S3qmuRn/nk9urwt2TwIEvtFJndfooAIVD8IquqNaxQGhbKo2bFSuxB2dz9OBd0V4+uhlYPvcb+iz8rvDtl4GT72j1psHILgDST87Sc8K2vKrHD1xpvKx6F1pngVLX8W4xC/i83ifsIpj2gvAH4SWC+f3auVOQ+/pDW4x6o/SEwdFZhmSBz3RljmSDfW1ZwLVx3/cJsZ3w7WdYuDc652/ZW/8zhMEnPpP7WIvRj0RGDA7DwbuMDvxuOEuIsUqFbJ0CPiF5hfVB5X7Ctx+fU9KP6AIwrIgdVps8cxku6pxX0yGIfGJG+IzBnu0wtgzA/g9s6yzWC0Jw1XJSOy4Ut8sevP6psI9ggfRhkePAIfEeob5O9F6A3un1e+wEnvO7HRe60x2AXaeAMF93EKzoCaFcUNeOBWLtSofsRkJcFAOjH9gzoMULBp37hQUYUzQWPXgbMCj+gPC8cKWYu/0+WUIC8Ldc2xaAkK7cbb1F+JnQo4x9KpKFYgIdfUubR4Qj2KwFoPzIDoI1vhS+b/DjRHEgmNWVpYQKlG3mvBK7ObsoZkzkJAzpB7DdOzc/1MH6fxaAZQb3Te/+fNg5CeN/GP1EW5NYRYegTE26UOKR0lOrV+cWgoVaS66V+2gLoai5I/mbXSS4rRvnOpULmEttW6Cc+Bstp3mlb5yx/5Z+Lvk//6XcsAK3m1S9C20psNmypcBmy5YCmy1Vr8DfR4wA0XwpYEkAAAAASUVORK5CYII=\" y=\"-6.488136\"/>\n   </g>\n   <g id=\"patch_38\">\n    <path d=\"M 404.420339 54.488136 \nL 404.420339 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_39\">\n    <path d=\"M 451.708475 54.488136 \nL 451.708475 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_40\">\n    <path d=\"M 404.420339 54.488136 \nL 451.708475 54.488136 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_41\">\n    <path d=\"M 404.420339 7.2 \nL 451.708475 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n  <g id=\"axes_9\">\n   <g id=\"patch_42\">\n    <path d=\"M 461.166102 54.488136 \nL 508.454237 54.488136 \nL 508.454237 7.2 \nL 461.166102 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g clip-path=\"url(#p4d61318710)\">\n    <image height=\"48\" id=\"image4606551f61\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"461.166102\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABHNCSVQICAgIfAhkiAAABHhJREFUaIHtmV1rXEUYx39JjF0SuwSlUbSlIWtraBqpFrFGi6URgq4vCEIR0doL0SJ44aWfwC8geiNeClXIhS1SEEHFSl+oJrY1QS0JymJNGmKk6Sbb3Xox/+fM7Ml2OWuEcnJ8YPk/M2fmnJl5XufZti64Toqp/WYvYK2U+g3c0uqEDmE7kBOfa/DM+KqwApQDPkQb828oOxLYKNwifBB4QvyoML9TzDb8cV9wMDkDX6jrG+HPwnngb/GtSiX1EmhL4ka7gXvF7xe+AmwvqnFYWOwX81Iw+0MHx0rwvmN/PebwY404AVwUvyBcIpkUUi+BRDawEbCzHRZu3wo8rsZDNtL80SRwTfxdDvaW4JJjC7MOXzjlMDxpk0QNb0bNJNF0A+YKczg1ArjdHt4J3Cr+rHBeFnvxAqzUr59NeEvd5GBA8h+oQcleISzjX9FsA9lQoRXgingTKzWgU7w9NP94FH+kW4V7gR7xy/Xv79IPvLtewAu4wo0pGxKoALK7KPjsm8YfjUW3HcIzcOV3x3bbxFIwftHB1ZrDMk6gIXXiBRxPSxJvwCYsETkQxoXn52DwBzUsJL8o7IHuT8Wb4d6Nl7dU6K9gSDxPim/oRpQNFVrBR8gp4XFgcEwNCw6vycm+vBmGJxxvUprH65+M/g81Z4P3h87CpGGn3EiFsiGBCs4OwHvHM8BnUuJnjqjzOYWh3negoKhc+MrhtU/gI42TwocSMFOx71gQg+b2kA0JgNc/8xazeJXmhPC0sJgD3lTjdX1pBPa/4fj39C652iX8ydt3aiTLRhNtoIP6vAhcxLT8KOqMQuZ5QKuLkqEiFJ507POfA7BLPvl7YEPwrVYoGyrUjs9VlEiyA+89saukJTJ8C7wt3tLqR4Fdjj3sJNAnCTw7FgXnSJXK1BcEmq0t1ZT4PmCJ5GbhEDB0mxoPxyZMTMC4AtmM+h4Yg6LOq7fX4Vt/ArD7MlS+dl3mMqv49CUugdC4m27AxNPJahXqB5+83SO0MHqayOcv/uQwvwE4pOUdcgtnm8YfhD1a7cKUX7Tdk2zBC7F2uMbUUuI4YJ7SXGfXqgZe/peILreKw5SXYfQDx+ctmBwQbiHKaEclgRKr8yMz8FClsiMB23U5xHJskLnRnbjSHTD0ncNJ/KUof1LMgPA+ompBm66g/TP+njQttPkdeDtY3xIwlV7GZ4tW9pgCdquKwm9CnTpPER1f31HhOP7Y+oRWo6mySsEbpRSN+hJdKVeAy+Kt8DQO9Cud3nNcneZWh4CR+x0/MqfOOVhUkmxWaXTWf6CqfH0arzLNyvHrW4WMKngVmhZW8ZeOskqE+961DuDVc2pYoXcQ8jr6vJ7N6U76CyBjt9LSObxmmtquyytlovJ6SFaryQF3iLfi22PCA0DhETWeFg5TX/sFcEkp14+A3Uq/FP6Iz4UsoDWygZY3EFL8kmMBuQdXBiJAy6HA14NmAzQ+TKuT/FuTPRVaC8X9+Fr+nTRKvQRa/p94LfRfnHicUi+B/zdwsyn1G/gHGeQsq/lBYtAAAAAASUVORK5CYII=\" y=\"-6.488136\"/>\n   </g>\n   <g id=\"patch_43\">\n    <path d=\"M 461.166102 54.488136 \nL 461.166102 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_44\">\n    <path d=\"M 508.454237 54.488136 \nL 508.454237 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_45\">\n    <path d=\"M 461.166102 54.488136 \nL 508.454237 54.488136 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_46\">\n    <path d=\"M 461.166102 7.2 \nL 508.454237 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n  <g id=\"axes_10\">\n   <g id=\"patch_47\">\n    <path d=\"M 517.911864 54.488136 \nL 565.2 54.488136 \nL 565.2 7.2 \nL 517.911864 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g clip-path=\"url(#pb5836fd4ab)\">\n    <image height=\"48\" id=\"imagec1fdad3dc2\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"517.911864\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABHNCSVQICAgIfAhkiAAABDRJREFUaIHtmUFoXFUUhr/MmDh2TKkp6qAIbccyQ4OUVkuLWotQUEPVTWMRqpsu2kUFN24Et24KdSW6Ebsr2ECxUldaraJSEK0gsaWtNBQkUpTEmDIhTnVx/z/35jlMJloYZiZn85/77nv3vXP/c86997y+VfA3HSy5dn/A/5XbWr0xn2kXgNulF4WrMm2AurCWXJvPXKsBc5lrdVqTlgzIE6nyx90LlKVXhBuF9yXPTiV4U/p14TXhVeAX6b8LZ4mGNpPecKE60C/dFg8QXeYB4WbhxqTvhnASmJHu2faYteQ+Yzr7zZjoDQYgzoJxlujfvwkdeMU7WaClqCktX4NJBYED1fGUJohcBpeS3mHA4tmbAn6VPpHBrX9C36Aa9wsLUPpZz4pGz16dyN7NBK03k2Ub4EHniOnwonBIWAGG7WvOqYPxK4cuByyoK5+MW8/gUtJ7LmSpEwIZQooEGBd+Cww7V+4SVljIm/doBStq+c0RZ7yVxSuV3mAg3UoMCPuJi5V92QF+CZhUhJecIzcTg+RSgKHv45jZvdYt3QvliDnbyeVuYnyuzfStIa4RJfvZJmC7dEX9BhngFTkrrRjR3S5kWgtE9qvCLcAO6Q8KzcA8gaFFsn4AeD7o+04AUH5Hz80tfwW2dDcD9s1BYIN0Z8WXgdXPqLE1uRFCNDsINnm0KnAwqDsU4bu/AKB0OiYCS55ePg/Y91MG1kl/Urj6BWC/GnZ4p408cUodPFSAh6W/HmBfYGDLaTijHj/mc8F/MsBiQ4rElDlsqx4n+pU2aXwnrBHPmTsbjfxUAE3AY6/A2PTioaaT9zdLp93pQhZb108SZA7UEmHFArggfCvAlenI2B3PSXnzBAw7BRwI0PdiwL3H2fZeUH/UHddZXMlY6hs7Vpoy4D36PPFAvpAeayzeVwCz8uP3iTvVR08FHJ0ATh4OjfV/qVcU7oLtYuBj9VxNxmgWAw0NyBajZomVhAuyqjoO7NVFxWRR7lI5BSfV5VNb8QcYeUONY6/q7VpAhqCs5FBS8i8Sa0S9W5UwE9MESgG+FFY/BLwSP6F8ejQkwZdmYO1n4dI53TKVNkzPI8q7E1DXNNsrc/x7i91IupsB+94MMQa+EVZ/gp1vq/GQlp/yswHPfMrIu2EtHTmeDOLzgKfZxdHxWCBo5O/NFrTuZsBSIx7czwvHgMIHQd/ms+XRjwLe9RocUu36kIsulwnRBOHYD3yuHc9FuKIeZ545WiuxtFzc9ebKrnSWuCTsORZwVN/FkSPwtNzJGzdKwFfSVRg6F0atfxLjW8dl/ljiwy0d70J9y/1H5oDKEWPR1UOX13cDo/59s0e4jkjZ2QDnRcQY8LW6vBudokcONMtmoJGkh38I/85chklLJtkSffqPzNdarQdZbokB7ZSOd6EVA9otKwa0W1YMaLd0vAH/ALIt9y4piRzoAAAAAElFTkSuQmCC\" y=\"-6.488136\"/>\n   </g>\n   <g id=\"patch_48\">\n    <path d=\"M 517.911864 54.488136 \nL 517.911864 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_49\">\n    <path d=\"M 565.2 54.488136 \nL 565.2 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_50\">\n    <path d=\"M 517.911864 54.488136 \nL 565.2 54.488136 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_51\">\n    <path d=\"M 517.911864 7.2 \nL 565.2 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"pcd8e0a022e\">\n   <rect height=\"47.288136\" width=\"47.288136\" x=\"7.2\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"p767374a98d\">\n   <rect height=\"47.288136\" width=\"47.288136\" x=\"63.945763\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"p529ff754ae\">\n   <rect height=\"47.288136\" width=\"47.288136\" x=\"120.691525\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"pc7affd4af1\">\n   <rect height=\"47.288136\" width=\"47.288136\" x=\"177.437288\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"pec8c9b862b\">\n   <rect height=\"47.288136\" width=\"47.288136\" x=\"234.183051\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"p9efdcdbc94\">\n   <rect height=\"47.288136\" width=\"47.288136\" x=\"290.928814\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"p728b23a716\">\n   <rect height=\"47.288136\" width=\"47.288136\" x=\"347.674576\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"p9117ebcde8\">\n   <rect height=\"47.288136\" width=\"47.288136\" x=\"404.420339\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"p4d61318710\">\n   <rect height=\"47.288136\" width=\"47.288136\" x=\"461.166102\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"pb5836fd4ab\">\n   <rect height=\"47.288136\" width=\"47.288136\" x=\"517.911864\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 720x720 with 10 Axes>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Class probabilities \n[0.09871688 0.11236461 0.09930012 0.10218297 0.09736711 0.09035161\n 0.09863356 0.10441593 0.09751708 0.09915014]\n<NDArray 10 @cpu(0)>\n"
 }
]
```

Now we can compute the likelihoods of an image, given the model. This is statistician speak for $p(x | y)$, i.e. how likely it is to see a particular image under certain conditions (such as the label). Our Naive Bayes model which assumed that all pixels are independent tells us that

$$p(\mathbf{x} | y) = \prod_{i} p(x_i | y)$$

Using Bayes' rule, we can thus compute $p(y | \mathbf{x})$ via

$$p(y | \mathbf{x}) = \frac{p(\mathbf{x} | y) p(y)}{\sum_{y'} p(\mathbf{x} | y')}$$

Let's try this ...

```{.python .input  n=3}
# Get the first test item
data, label = mnist_test[0]
data = data.reshape((784,1))

# Compute the per pixel conditional probabilities
xprob = (px * data + (1-px) * (1-data))
# Take the product
xprob = xprob.prod(0) * py
print('Unnormalized Probabilities', xprob)
# Normalize
xprob = xprob / xprob.sum()
print('Normalized Probabilities', xprob)
```

This went horribly wrong! To find out why, let's look at the per pixel probabilities. They're typically numbers between $0.001$ and $1$. We are multiplying $784$ of them. At this point it is worth mentioning that we are calculating these numbers on a computer, hence with a fixed range for the exponent. What happens is that we experience *numerical underflow*, i.e. multiplying all the small numbers leads to something even smaller until it is rounded down to zero. At that point we get division by zero with `nan` as a result.

To fix this we use the fact that $\log a b = \log a + \log b$, i.e. we switch to summing logarithms. This will get us unnormalized probabilities in log-space. To normalize terms we use the fact that

$$\frac{\exp(a)}{\exp(a) + \exp(b)} = \frac{\exp(a + c)}{\exp(a + c) + \exp(b + c)}$$

In particular, we can pick $c = -\max(a,b)$, which ensures that at least one of the terms in the denominator is $1$.

```{.python .input  n=4}
logpx = nd.log(px)
logpxneg = nd.log(1-px)
logpy = nd.log(py)

def bayespost(data):
    # We need to incorporate the prior probability p(y) since p(y|x) is
    # proportional to p(x|y) p(y)
    logpost = logpy.copy()
    logpost += (logpx * data + logpxneg * (1-data)).sum(0)
    # Normalize to prevent overflow or underflow by subtracting the largest
    # value
    logpost -= nd.max(logpost)
    # Compute the softmax using logpx
    post = nd.exp(logpost).asnumpy()
    post /= np.sum(post)
    return post

fig, figarr = plt.subplots(2, 10, figsize=(10, 3))

# Show 10 images
ctr = 0
for data, label in mnist_test:
    x = data.reshape((784,1))
    y = int(label)

    post = bayespost(x)

    # Bar chart and image of digit
    figarr[1, ctr].bar(range(10), post)
    figarr[1, ctr].axes.get_yaxis().set_visible(False)
    figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[0, ctr].axes.get_xaxis().set_visible(False)
    figarr[0, ctr].axes.get_yaxis().set_visible(False)
    ctr += 1

    if ctr == 10:
        break

plt.show()
```

As we can see, this classifier works pretty well in many cases. However, the second last digit shows that it can be both incompetent and overly confident of its incorrect estimates. That is, even if it is horribly wrong, it generates probabilities close to 1 or 0. Not a classifier we should use very much nowadays any longer. To see how well it performs overall, let's compute the overall accuracy of the classifier.

```{.python .input  n=5}
# Initialize counter
ctr = 0
err = 0

for data, label in mnist_test:
    ctr += 1
    x = data.reshape((784,1))
    y = int(label)

    post = bayespost(x)
    if (post[y] < post.max()):
        err += 1

print('Naive Bayes has an error rate of', err/ctr)
```

Modern deep networks achieve error rates of less than 0.01. While Naive Bayes classifiers used to be popular in the 80s and 90s, e.g. for spam filtering, their heydays are over. The poor performance is due to the incorrect statistical assumptions that we made in our model: we assumed that each and every pixel are *independently* generated, depending only on the label. This is clearly not how humans write digits, and this wrong assumption led to the downfall of our overly naive (Bayes) classifier. Time to start building Deep Networks.

## Summary

* Naive Bayes is an easy to use classifier that uses the assumption
  $p(\mathbf{x} | y) = \prod_i p(x_i | y)$.
* The classifier is easy to train but its estimates can be very wrong.
* To address overly confident and nonsensical estimates, the
  probabilities $p(x_i|y)$ are smoothed, e.g. by Laplace
  smoothing. That is, we add a constant to all counts.
* Naive Bayes classifiers don't exploit any correlations between
  observations.

## Exercises

1. Design a Naive Bayes regression estimator where $p(x_i | y)$ is a normal distribution.
1. Under which situations does Naive Bayes work?
1. An eyewitness is sure that he could recognize the perpetrator with 90% accuracy, if he were to encounter him again.
   * Is this a useful statement if there are only 5 suspects?
   * Is it still useful if there are 50?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2320)

![](../img/qr_naive-bayes.svg)
