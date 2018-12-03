# Preface

Deep Learning has taken machine learning, statistics and artificial
intelligence by storm. The past five years have witnessed tremendous
progress in computer vision, natural language processing,
reinforcement learning, and statistical modeling. As a result we are
now able to build cars that drive themselves, systems that converse
with humans in an almost natural manner, that learn how to play games
and that can deal with large amounts of natural data, from medical
science to astrophysics. All of this has led to a great deal of
excitement and demand for deep learning.

## Code, Math and HTML

Technology is most valuable when it is well understood, actionable and
up to date. Understanding is needed to allow its user to modify,
extend and apply it efficiently. Understanding per se, however does
not lead to success, unless it is paired with the ability to apply the
knowledge. Lastly, with rapid progress comes the opportunity and need
to incorporate new ideas in one's repertoire. 

Deep Learning is one such technology and this book is an attempt to
make it universally accessible, for engineers and researchers
alike. We started this book project in June 2017 when we needed to
explain the new Gluon MXNet interface to users. At the time there were
no resources that were (1) up to date, (2) that covered the full
breadth of modern machine learning with some degree of technical depth
and (3) that interleave an engaging textbook with runnable code. There
were plenty of code examples for how to use a particular deep learning
framework (e.g. how to deal with matrices in TensorFlow) or for
implementing a particular technique (e.g. code snippets for LeNet) in
the form of blog posts or on GitHub. However these examples typically
focused on the *how* rather than also addressing the question of *why*
certain algorithmic decisions are made. Whenever such resources
existed, e.g. in the form of [Distill](http://distill.pub), they only
covered parts of deep learning, often lacking associated code. At the
same time, textbooks such as that of
[Goodfellow, Bengio and Courville, 2016](https://www.deeplearningbook.org/)
offered an excellent survey of the theoretical concepts without much
indication of how to implement them. Others still were hidden behind
the paywall of commercial course providers.

We set out to create a resource that was (1) freely available for
everyone, (2) of sufficient technical depth to help a reader
understand *why* something worked, (3) that included runnable code to
show readers *how* to solve problems in practice, and (4) that allowed
for rapid updates, both by us, and also by the community at
large. This is complemented by (5) a [forum](http://discuss.mxnet.io)
to discuss technical details and to answer questions.

These goals were often in conflict regarding how to generate and
deliver the material: Equations, theorems, and citations are best
managed and laid out in LaTeX, code is best described in Python,
webpages are native in HTML and JavaScript, moreover they offer
efficient search and linking. Furthermore, we want the content to be
accessible both as executable code, as a book, as downloadable PDF,
and on the internet as a website. At present there exist no tools and
no workflow that is well suited to these demands, hence we had to
assemble our own. They are described in detail in the
[appendix](../chapter_appendix/how-to-contribute.md). We settled on
Github to share the source and to allow for edits, Jupyter notebooks
to mix code and text, Sphinx as rendering engine to generate multiple
outputs and Discourse for the forum. This provides a good compromise,
albeit not a perfect one for high quality layout, e.g. of figures and
citations. We believe that this might be the first book published
using such an integrated workflow.

## Organization

Aside from a few (optional) notebooks providing a crash course in the basic mathematical background, each subsequent notebook introduces both a reasonable number of new concepts and provides a single self-contained working example, using a real dataset. 
This presents an organizational challenge. Some models might logically
be grouped together in a single notebook.  And some ideas might be
best taught by executing several models in succession.  On the other
hand, there's a big advantage to adhering to a policy of *1 working
example, 1 notebook*: This makes it as easy as possible for you to
start your own research projects by plagiarising our code. Just copy a
single notebook and start modifying it.

We will interleave the runnable code with background material as
needed.  In general, we will often err on the side of making tools
available before explaining them fully (and we will follow up by
explaining the background later).  For instance, we might use
*stochastic gradient descent* before fully explaining why it is useful
or why it works.  This helps to give practitioners the necessary
ammunition to solve problems quickly, at the expense of requiring the
reader to trust us with some decisions, at least in the short term.

Throughout, we'll be working with the MXNet library, which has the
rare property of being flexible enough for research while being fast
enough for production.  This book will teach deep learning concepts
from scratch.  Sometimes, we want to delve into fine details about the
models that are hidden from the user by ``Gluon``'s advanced features.
This comes up especially in the basic tutorials, where we want you to
understand everything that happens in a given layer.  In these cases,
we generally present two versions of the example: one where we
implement everything from scratch, relying only on NDArray and
automatic differentiation, and another where we show how to do things
succinctly with ``Gluon``.  Once we've taught you how a layer works,
we can just use the ``Gluon`` version in subsequent tutorials.

## Learning by Doing

Many textbooks teach a series of topics, each in exhaustive detail. For example, Chris Bishop's excellent textbook, [Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738), teaches each topic so thoroughly, that getting to the chapter on linear regression requires a non-trivial amount of work. For beginners this actually limited the book's usefulness as an introductory text, while experts will love it precisely for its thoroughness. But perhaps the traditional textbook aproach is not the easiest way to get started in the first place. 

Instead, in this book, we'll teach most concepts just in time. For the
fundamental preliminaries like linear algebra and probability, we'll
provide a brief crash course from the outset, but we want you to taste
the satisfaction of training your first model before worrying about
exotic probability distributions.

If you're ready to get started, head over to [the introduction](../chapter_introduction/index.rst). For everything else [open an issue on Github](https://github.com/diveintodeeplearning/d2l-en). 

## Thanks

We are indebted to the hundreds of contributors for both the English draft and subsequently the Chinese version. They helped improve the content and offered valuable feedback. Moreover, we thank Amazon Web Services for its generous support in writing this book. Without the available time, resources, discussions with colleagues, and continuous encouragement this book would not have happened. 
