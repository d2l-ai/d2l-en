# From MLPs to Convnets

So far we learned the basics of designing Deep Networks. Indeed, for someone dealing only with generic data, the previous sections are probably sufficient to train and deploy such a network sufficiently. There is one caveat, though - just like most problems in statistics, networks with many parameters either require a lot of data or a lot of regularization. As a result we cannot hope to design sophisticated models in most cases. 

For instance, consider the seemingly task of distinguishing cats from dogs. We decide to use a good camera and take 1 megapixel photos to ensure that we can really distinguish both species accurately. This means that the *input* into a network has 1 million dimensions. Even an aggressive reduction to 1,000 dimensions after the first layer means that we need $10^9$ parameters. Unless we have copious amounts of data (billions of images of cats and dogs), this is mission impossible. Add in subsequent layers and it is clear that this approach is infeasible. 

The avid reader might object to the rather absurd implications of this argument by stating that 1 megapixel resolution is not necessary. But even if we reduce it to a paltry 100,000 pixels, that's still $10^8$ parameters. A corresponding number of images of training data is beyond the reach of most statisticians. In fact, the number exceeds the population of dogs and cats in all but the largest countries! Yet both humans and computers are able to distinguish cats from dogs quite well, often after only a few hundred images. This seems to contradict our conclusions above. There is clearly something wrong in the way we are approaching the problem. Let's find out.

## Invariances

Imagine that you want to detect an object in an image. It is only reasonable to assume that the location of the object shouldn't matter too much to determine whether the object is there. We should assume that we would recognize an object wherever it is in an image. This is true within reason - pigs usually don't fly and planes usually don't swim. Nonetheless, we would still recognize a flying pig, albeit possibly after a double-take. This fact manifests itself e.g. in the form of the children's game 'Where is Waldo'. In it, the goal is to find a boy with red and white striped clothes, a striped hat and black glasses within a panoply of activity in an image. Despite the rather characteristic outfit this tends to be quite difficult, due to the large amount of confounders. The image below, on the other hand, makes the problem particularly easy. 

![](../img/waldo.jpg)


There are two key principles that we can deduce from this slightly frivolous reasoning:

1. Object detectors should work the same regardless of where in the image an object can be found. In other words, the 'waldoness' of a location in the image can be assessed (in first approximation) without regard of the position within the image. (Translation Invariance)
1. Object detection can be answered by considering only local information. In other words, the 'waldoness' of a location can be assessed (in first approximation) without regard of what else happens in the image at large distances. (Locality)

Let's see how this translates into mathematics. 

## Constraining the MLP

In the following we will treat images and hidden layers as two-dimensional arrays. I.e. $x[i,j]$ and $h[i,j]$ denote the position $(i,j)$ in an image. Consequently we switch from weight matrices to four-dimensional weight tensors. In this case a dense layer can be written as follows:

$$h[i,j] = \sum_{k,l} W[i,j,k,l] x[k,l] = 
\sum_{a, b} V[i,j,a,b] \cdot x[i+a,j+b]$$

The switch from $W$ to $V$ is entirely cosmetic (for now) since there is a one to one correspondence between coefficients in both tensors. We simply re-index the subscripts $(k,l)$ such that $k = i+a$ and $l = j+b$. In other words we set $V[i,j,a,b] = W[i,j,i+a, j+b]$. The indices $a, b$ run over both positive and negative offsets, covering the entire image. For any given location $(i,j)$ in the hidden layer $h[i,j]$ we compute its value by summing over pixels in $x$, centered around $(i,j)$ and weighted by $V[i,j,a,b]$. 

Now let's invoke the first principle we established above - *translation invariance*. This implies that a shift in the inputs $x$ should simply lead to a shift in the activations $h$. This is only possible if $V$ doesn't actually depend on $(i,j)$, that is, we have $V[i,j,a,b] = V[a,b]$. As a result we can simplify the definition for $h$.

$$h[i,j] = \sum_{a, b} V[a,b] \cdot x[i+a,j+b]$$

This is a convolution! We are effectively weighting pixels $(i+a, j+b)$ in the vicinity of $(i,j)$ with coefficients $V[a,b]$ to obtain the value $h[i,j]$. Note that $V[a,b]$ needs a lot fewer coefficients than $V[i,j,a,b]$. For a 1 megapixel image it has at most 1 million coefficients. This is 1 million fewer parameters since it no longer depends on the location within the image. We have made significant progress! 

Now let's invoke the second principle - *locality*. In the problem of detecting Waldo we shouldn't have to look very far away from $(i,j)$ in order to glean relevant information to assess what is going on at $h[i,j]$. This means that outside some range $|a|, |b| > \Delta$ we should set $V[a,b] = 0$. Equivalently we can simply rewrite $h[i,j]$ as

$$h[i,j] = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} V[a,b] \cdot x[i+a,j+b]$$

This, in a nutshell is the convolutional layer. The difference to the fully connected network is dramatic. While previously we might have needed $10^8$ or more coefficients, we now only need $O(\Delta^2)$ terms. The price that we pay for this drastic simplification is that our network will be translation invariant and that we are only able to take local information into account. 

## Waldo Revisited

Let's see what this looks like if we want to build an improved Waldo detector. The convolutional layer picks windows of a given size and weighs intensities according to the mask $V$. We expect that wherever the 'waldoness' is highest, we will also find a peak in the hidden layer activations.  

![](../img/waldo-mask.jpg)

There's just a problem with this approach: so far we blissfully ignored that images consist of 3 channels - red, green and blue. In reality images are thus not two-dimensional objects but three-dimensional tensors, e.g. of $1024 \times 1024 \times 3$ pixels. We thus index $\mathbf{x}$ as $x[i,j,k]$. The convolutional mask has to adapt accordingly. Instead of $V[a,b]$ we now have $V[a,b,c]$. 

The last flaw in our reasoning is that this approach generates only one set of activations. This might not be great if we want to detect Waldo in several steps. We might need edge detectors, detectors for different colors, etc.; In short, we want to retain some information about edges, color gradients, combinations of colors, and a great many other things. An easy way to address this is to allow for *output channels*. We can take care of this by adding a fourth coordinate to $V$ via $V[a,b,c,d]$. Putting all together we have:

$$h[i,j,k] = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c V[a,b,c,k] \cdot x[i+a,j+b,c]$$

This is the definition of a convolutional neural network layer. There are still many operations that we need to address. For instance, we need to figure out how to combine all the activations to a single output (e.g. whether there's a Waldo in the image). We also need to decide how to compute things efficiently, how to combine multiple layers, and whether it is a good idea to have many narrow or a few wide layers. All of this will be addressed in the remainder of the chaper. For now we can bask in the glory having understood why convolutions exist in principle. 

## Summary

* Translation invariance in images implies that all patches of an image will be treated in the same manner.
* Locality means that only a small neighborhood of pixels will be used for computation.
* Channels on input and output allows for meaningful feature analysis.

## Problems

* Assume that the size of the convolution mask is $\Delta = 0$. Show that in this case the convolutional mask implements an MLP independently for each set of channels. 
* Why might translation invariance not be a good idea after all? Does it make sense for pigs to fly?
* What happens at the boundary of an image?
* Derive an analogous convolutional layer for audio.
* What goes wrong when you apply the above reasoning to text? Hint - what is the structure of language?



