
# Preface

If you're a reasonable person, you might ask, "what is *mxnet-the-straight-dope*?" You might also ask, "why does it have such an ostentatious name?" Speaking to the former question, *mxnet-the-straight-dope* is an attempt to create a new kind of educational resource for deep learning. Our goal is to leverage the strengths of Jupyter notebooks to present prose, graphics, equations, and (importantly) code together in one place. If we're successful, the result will be a resource that could be simultaneously a book, course material, a prop for live tutorials, and a resource for plagiarising (with our blessing) useful code. To our knowledge, few available resources aim to teach either (1) the full breadth of concepts in modern machine learning or (2) interleave an engaging textbook with runnable code. We'll find out by the end of this venture whether or not that void exists for a good reason.

Regarding the name, we are cognizant that the machine learning community and the ecosystem in which we operate have lurched into an absurd place. In the early 2000s, comparatively few tasks in machine learning had been conquered, but we felt that we understood *how* and *why* those models worked (with some caveats). By contrast, today's machine learning systems are extremely powerful and *actually work* for a growing list of tasks, but huge open questions remain as to precisely *why* they are so effective.  

This new world offers enormous opportunity, but has also given rise to considerable buffoonery. Research preprints like [the arXiv](http://arxiv.org) are flooded by clickbait, AI startups have sometimes received overly optimistic valuations, and the blogosphere is flooded with thought leadership pieces written by marketers bereft of any technical knowledge. Amid the chaos, easy money, and lax standards, we believe it's important not to take our models or the environment in which they are worshipped too seriously. Also, in order to both explain, visualize, and code the full breadth of models that we aim to address, it's important that the authors do not get bored while writing. 

## Organization

At present, we're aiming for the following format: aside from a few (optional) notebooks providing a crash course in the basic mathematical background, each subsequent notebook will both:

1. Introduce a reasonable number (perhaps one) of new concepts
2. Provide a single self-contained working example, using a real dataset

This presents an organizational challenge. Some models might logically be grouped together in a single notebook. 
And some ideas might be best taught by executing several models in succession. 
On the other hand, there's a big advantage to adhering to a policy of *1 working example, 1 notebook*:
This makes it as easy as possible for you to start your own research projects 
by plagiarising our code. Just copy a single notebook and start modifying it.

We will interleave the runnable code with background material as needed. 
In general, we will often err on the side of making tools available before explaining them fully 
(and we will follow up by explaining the background later). 
For instance, we might use *stochastic gradient descent* 
before fully explaining why it is useful or why it works. 
This helps to give practitioners the necessary ammunition to solve problems quickly, 
at the expense of requiring the reader to trust us with some decisions, at least in the short term. 
Throughout, we'll be working with the MXNet library, 
which has the rare property of being flexible enough for research 
while being fast enough for production. 
Our more advanced chapters will mostly rely 
on MXNet's new high-level imperative interface ``gluon``. 
Note that this is not the same as ``mxnet.module``, 
an older, symbolic interface supported by MXNet. 

This book will teach deep learning concepts from scratch. 
Sometimes, we'll want to delve into fine details about the models 
that are hidden from the user by ``gluon``'s advanced features. 
This comes up especially in the basic tutorials, 
where we'll want you to understand everything that happens in a given layer. 
In these cases, we'll generally present two versions of the example: 
one where we implement everything from scratch, 
relying only on NDArray and automatic differentiation, 
and another where we show how to do things succinctly with ``gluon``. 
Once we've taught you how a layer works, 
we can just use the ``gluon`` version in subsequent tutorials.

## Learning by doing

Many textbooks teach a series of topics, each in exhaustive detail. For example, Chris Bishop's excellent textbook, [Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738), teaches each topic so thoroughly, that getting to the chapter on linear regression requires a non-trivial amount of work. When I (Zack) was first learning machine learning, this actually limited the book's usefulness as an introductory text. When I rediscovered it a couple years later, I loved it precisely for its thoroughness, and I hope you check it out after working through this material! But perhaps the traditional textbook aproach is not the easiest way to get started in the first place. 

Instead, in this book, we'll teach most concepts just in time. For 
the fundamental preliminaries like linear algebra and probability, 
we'll provide a brief crash course from the outset, 
but we want you to taste the satisfaction of training your first model 
before worrying about exotic probability distributions. 

## Next steps

If you're ready to get started, head over to [the introduction](../chapter01_crashcourse/introduction.ipynb) or go straight to [our basic primer on NDArray](./ndarray.ipynb), MXNet's workhorse data structure.


For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
