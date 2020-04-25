# From Dense Layers to Convolutions

The models that we have discussed so far 
remain (to this day) appropriate options
when we are dealing with tabular data.
By *tabular*, we mean that the data consists
of rows corresponding to examples
and columns corresponding to features.
With tabular data, we might anticipate 
that the patterns we seek could involve 
interactions among the features,
but we do not assume any structure *a priori* 
concerning how the features interact.

Sometimes, we truly lack knowledge to guide
the construction of craftier architectures.
In these cases, a multilayer perceptron 
may be the best that we can do.
However, for high-dimensional perceptual data,
these *structure-less* networks can grow unwieldy.

For instance, let's return to our running example
of distinguishing cats from dogs.
Say that we do a thorough job in data collection,
collecting an annotated dataset of 1-megapixel photographs.
This means that each input to the network has *1 million dimensions*.
Even an aggressive reduction to *1,000 hidden dimensions*
would require a *dense* (fully connected) layer 
characterized by $10^9$ parameters.
Unless we have lots of GPUs, a talent 
for distributed optimization,
and an extraordinary amount of patience,
learning the parameters of this network 
may turn out to be infeasible.

A careful reader might object to this argument
on the basis that 1 megapixel resolution may not be necessary.
However, while we might be able 
to get away with 100,000 pixels,
our hidden layer of size $1000$ grossly underestimated 
the number of hidden nodes that it takes 
to learn good representations of images,
so a practical system will still require billions of parameters.
Moreover, learning a classifier by fitting so many parameters
might require collecting an enormous dataset.
And yet today both humans and computers are able 
to distinguish cats from dogs quite well, 
seemingly contradicting these intuitions.
That is because images exhibit rich structure
that can be exploited by humans 
and machine learning models alike.
Convolutional neural networks are one creative way
that machine learning has embraced for exploiting
some of the known structure in natural images.


## Invariances

Imagine that you want to detect an object in an image.
It seems reasonable that whatever method 
we use to recognize objects should not be overly concerned 
with the precise *location* of the object in the image.
Ideally, our system should exploit this knowledge.
Pigs usually do not fly and planes usually do not swim.
Nonetheless, we should still recognize
a pig were one to appear at the top of the image.

We can draw some inspiration here 
from the children's game 'Where's Waldo'
(depicted in :numref:`img_waldo`).
The game consists of a number of chaotic scenes 
bursting with activity.
Waldo shows up somewhere in each,
typically lurking in some unlikely location.
The reader's goal is to locate him.
Despite his characteristic outfit, 
this can be surprisingly difficult,
due to the large number of distractions.
However, *what Waldo looks like* 
does not depend upon *where Waldo is located*.
We could sweep the image with a Waldo detector 
that could assign a score to each patch,
indicating the likelihood that the patch contains Waldo.
CNNs systematize this idea of spatial invariance,
exploiting it to learn useful representations 
with few parameters.

![Image via Walker Books](../img/where-wally-walker-books.jpg)
:width:`400px`
:label:`img_waldo`


We can now make these intuitions more concrete
by enumerating a few desiderata to guide our design 
of a neural network architecture suitable for computer vision:

1. In the earliest layers, our network 
    should respond similarly to the same patch, 
    regardless of where it appears in the image (translation invariance).
1. The earliest layers of the network should focus on local regions, 
   without regard for the contents of the image in distant regions (locality). 
   Eventually, these local representations can be aggregated
   to make predictions at the whole image level.

Let us see how this translates into mathematics.


## Constraining the MLP

To start off, we can consider an MLP 
with $h \times w$ images as inputs
(represented as matrices in math, and as 2D arrays in code),
and hidden representations **similarly organized
as** $h \times w$ **matrices / 2D arrays**.
Let that sink in, we now conceive of not only the inputs but 
also the hidden representations as possessing spatial structure.
Let $x[i, j]$ and $h[i, j]$ denote pixel location $(i, j)$
in the input image and hidden representation, respectively.
Consequently, to have each of the $hw$ hidden nodes 
receive input from each of the $hw$ inputs,
we would switch from using weight matrices
(as we did previously in MLPs)
to representing our parameters
as four-dimensional weight tensors.


We could formally express this dense layer as follows:

$$h[i, j] = u[i, j] + \sum_{k, l} W[i, j, k, l] \cdot x[k, l] =  u[i, j] +
\sum_{a, b} V[i, j, a, b] \cdot x[i+a, j+b].$$

The switch from $W$ to $V$ is entirely cosmetic (for now)
since there is a one-to-one correspondence
between coefficients in both tensors.
We simply re-index the subscripts $(k, l)$
such that $k = i+a$ and $l = j+b$.
In other words, we set $V[i, j, a, b] = W[i, j, i+a, j+b]$.
The indices $a, b$ run over both positive and negative offsets,
covering the entire image.
For any given location $(i, j)$ in the hidden layer $h[i, j]$,
we compute its value by summing over pixels in $x$,
centered around $(i, j)$ and weighted by $V[i, j, a, b]$.

Now let us invoke the first principle 
established above: *translation invariance*.
This implies that a shift in the inputs $x$
should simply lead to a shift in the activations $h$.
This is only possible if $V$ and $u$ do not actually depend on $(i, j)$,
i.e., we have $V[i, j, a, b] = V[a, b]$ and $u$ is a constant.
As a result, we can simplify the definition for $h$.

$$h[i, j] = u + \sum_{a, b} V[a, b] \cdot x[i+a, j+b].$$

This is a convolution!
We are effectively weighting pixels $(i+a, j+b)$
in the vicinity of $(i, j)$ with coefficients $V[a, b]$
to obtain the value $h[i, j]$.
Note that $V[a, b]$ needs many fewer coefficients than $V[i, j, a, b]$. 
For a 1 megapixel image, it has at most 1 million coefficients. 
This is 1 million fewer parameters since it 
no longer depends on the location within the image. 
We have made significant progress!

Now let us invoke the second principle---*locality*.
As motivated above, we believe that we should not have
to look very far away from $(i, j)$
in order to glean relevant information
to assess what is going on at $h[i, j]$.
This means that outside some range $|a|, |b| > \Delta$,
we should set $V[a, b] = 0$.
Equivalently, we can rewrite $h[i, j]$ as

$$h[i, j] = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} V[a, b] \cdot x[i+a, j+b].$$

This, in a nutshell, is a convolutional layer.
When the local region (also called a *receptive field*) is small,
the difference as compared to a fully-connected network can be dramatic.
While previously, we might have required billions of parameters
to represent just a single layer in an image-processing network,
we now typically need just a few hundred, without 
altering the dimensionality of either 
the inputs or the hidden representations.
The price paid for this drastic reduction in parameters
is that our features are now translation invariant
and that our layer can only incorporate local information,
when determining the value of each hidden activation.
All learning depends on imposing inductive bias.
When that bias agrees with reality,
we get sample-efficient models
that generalize well to unseen data.
But of course, if those biases do not agree with reality,
e.g., if images turned out not to be translation invariant,
our models might struggle even to fit our training data.


## Convolutions

Before going further, we should briefly review 
why the above operation is called a *convolution*.
In mathematics, the convolution between two functions,
say $f, g: \mathbb{R}^d \to R$ is defined as

$$[f \circledast g](x) = \int_{\mathbb{R}^d} f(z) g(x-z) dz.$$

That is, we measure the overlap between $f$ and $g$
when both functions are shifted by $x$ and "flipped".
Whenever we have discrete objects, the integral turns into a sum.
For instance, for vectors defined on $\ell_2$, i.e.,
the set of square summable infinite dimensional vectors
with index running over $\mathbb{Z}$ we obtain the following definition.

$$[f \circledast g](i) = \sum_a f(a) g(i-a).$$

For two-dimensional arrays, we have a corresponding sum
with indices $(i, j)$ for $f$ and $(i-a, j-b)$ for $g$ respectively.
This looks similar to definition above, with one major difference.
Rather than using $(i+a, j+b)$, we are using the difference instead.
Note, though, that this distinction is mostly cosmetic
since we can always match the notation by using $\tilde{V}[a, b] = V[-a, -b]$
to obtain $h = x \circledast \tilde{V}$.
Our original definition more properly 
describes a *cross correlation*.
We will come back to this in the following section.


## Waldo Revisited

Returning to our Waldo detector, let's see what this looks like. 
The convolutional layer picks windows of a given size
and weighs intensities according to the mask $V$, as demonstrated in :numref:`fig_waldo_mask`.
We might aim to learn a model so that 
wherever the "waldoness" is highest,
we should find a peak in the hidden layer activations.

![Find Waldo.](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

There's just one problem with this approach.
So far, we blissfully ignored that images consist
of 3 channels: red, green and blue.
In reality, images are not two-dimensional objects
but rather $3^{\mathrm{rd}}$ order tensors,
characterized by a height, width, and *channel*,
e.g., with shape $1024 \times 1024 \times 3$ pixels.
While the first two of these axes concern spatial relationships,
the $3^{\mathrm{rd}}$ can be regarded as assigning
a multidimensional representation *to each pixel location*.

We thus index $\mathbf{x}$ as $x[i, j, k]$.
The convolutional mask has to adapt accordingly.
Instead of $V[a, b]$, we now have $V[a, b, c]$.

Moreover, just as our input consists of a $3^{\mathrm{rd}}$ order tensor,
it turns out to be a good idea to similarly formulate
our hidden representations as $3^{\mathrm{rd}}$ order tensors.
In other words, rather than just having a single activation
corresponding to each spatial location,
we want an entire vector of hidden activations
corresponding to each spatial location.
We could think of the hidden representation as comprising
a number of 2D grids stacked on top of each other.
As in the inputs, these are sometimes called *channels*.
They are also sometimes called *feature maps*,
as each provides a set of spatialized set
of learned features to the subsequent layer.
Intuitively, you might imagine that at lower layers,
some channels could become specialized to recognize edges,
others to recognize textures, etc.
To support multiple channels in both inputs and hidden activations,
we can add a fourth coordinate to $V$: $V[a, b, c, d]$. 
Putting everything together we have:

$$h[i, j, k] = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c V[a, b, c, k] \cdot x[i+a, j+b, c].$$

This is the definition of a convolutional neural network layer.
There are still many operations that we need to address.
For instance, we need to figure out how to combine all the activations
to a single output (e.g., whether there is a Waldo *anywhere* in the image).
We also need to decide how to compute things efficiently,
how to combine multiple layers, 
appropriate activation functions,
and how to make reasonable design choices
to yield networks that are effective in practice.
We turn to these issues in the remainder of the chapter.


## Summary

* Translation invariance in images implies that all patches of an image will be treated in the same manner.
* Locality means that only a small neighborhood of pixels will be used to compute the corresponding hidden activation.
* Channels on input and output allow our model to capture multiple aspects of an image  at each spatial location.

## Exercises

1. Assume that the size of the convolution mask is $\Delta = 0$. 
   Show that in this case the convolutional mask 
   implements an MLP independently for each set of channels.
1. Why might translation invariance not be a good idea after all? 
   When might it not make sense to allow for pigs to fly?
1. What problems must we deal with when deciding how 
   to treat activations corresponding to pixel locations
   at the boundary of an image?
1. Describe an analogous convolutional layer for audio.
1. Do you think that convolutional layers might also be applicable for text data?
   Why or why not?
1. Prove that $f \circledast g = g \circledast f$.

## [Discussions](https://discuss.mxnet.io/t/2348)

![](../img/qr_why-conv.svg)
