# Preface

Just a few years ago, there were no legions of deep learning scientists
developing intelligent products and services at major companies and startups.
When the youngest of us (the authors) entered the field,
machine learning didn't command headlines in daily newspapers.
Our parents had no idea what machine learning was,
let alone why we might prefer it to a career in medicine or law.
Machine learning was a forward-looking academic discipline
with a narrow set of real-world applications.
And those applications, e.g. speech recognition and computer vision,
required so much domain knowledge that they were often regarded
as separate areas entirely for which machine learning was one small component.
Neural networks, the antecedents of the deep learning models
that we focus on in this book, were regarded as outmoded tools.


In just the past five years, deep learning has taken the world by surprise,
driving rapid progress in fields as diverse as computer vision,
natural language processing, automatic speech recognition,
reinforcement learning, and statistical modeling.
With these advances in hand, we can now build cars that drive themselves
(with increasing autonomy), smart reply systems that anticipate mundane replies,
helping people dig out from mountains of email,
and software agents that dominate the world's best humans at board games like Go,
a featonce deemed to be decades away.
Already, these tools are exerting a widening impact,
changing the way movies are made, diseases are diagnosed,
and playing a growing role in basic sciences, from astrophysics to biology.
This book represents our attempt to make deep learning approachable,
teaching you both the *concepts*, the *context*, and the *code*.


## One Medium Combining Code, Math, and HTML


For any computing technology to reach its full impact,
it must well-understood, well-documented, and supported by
mature, well-maintained tools.
The key ideas should be clearly distilled,
minimizing the onboarding time needing to bring new practitioners up to date.
Mature libraries should automate common tasks,
and exemplar code should make it easy for practitioners
to modify, apply, and extend common applications to suit their needs.
Take dynamic web applications as an example.
Despite a large number of companies, like Amazon,
developing successful database-driven web applications in the 1990s,
the full potential of this technology to aid creative entrepreneurs
has only been realized over the past ten years,
owing to the development of powerful, well-documented frameworks.



Realizing deep learning presents unique challenges because
any single application brings together various disciplines.
Applying deep learning requires simultaneously understanding
(i) the motivations for casting a problem in a particular way,
(ii) the mathematics of a given modeling approach,
(iii) the optimization algorithms for fitting the models to data,
(iv) and the engineering required to train models efficiently,
navigating the pitfalls of numerical computing and getting the most
out of available hardware.
Teaching both the critical thinking skills required to formulate problems,
the mathmematics to solve them, and the software tools to implement those
solutions all in one place presents formidable challenges.
Our goal in this book is to present a unified resource
to bring would-be practitioners up to speed.



We started this book project in July 2017
when we needed to explain MXNet's (then new) Gluon interface to our users.
At the time, there were no resources that were simultaneously
(1) up to date, (2) covered the full breadth of modern machine learning
with anything resembling of technical depth,
and (3) interleaved the exposition one expects from an egaging textbook
with the clean runnable code one seeks in hands-on tutorials.
We found plenty of code examples for how to use a given deep learning
framework (e.g. how to do basic numerical computing with matrices in TensorFlow)
or for implementing particular techniques (e.g. code snippets for LeNet, AlexNet, ResNets, etc)
in the form of blog posts or on GitHub.
However, these examples typically focused on
*how* to implement a given approach,
but left out the discussion of *why* certain algorithmic decisions are made.
While sporadic topics have been convered in blog posts,
e.g. on the website [Distill](http://distill.pub) or personal blogs,
they only covered selected topics in deep learning, and often lacked associated code.
One the other hand, while several textbooks have emerged,
most notably [Goodfellow, Bengio and Courville, 2016](https://www.deeplearningbook.org/),
which offers an excellent survey of the concepts behind deep learning,
these resources don't marry the descriptions to realizations of the concepts in code,
sometimes leaving readers clueless as to how to implement them.
Moreover, too many resources are hidden behind the paywalls of commercial course providers.

We set out to create a resource that could
(1) be freely available for everyone,
(2) offer sufficient technical depth to provide a starting point on the path
to actually becoming an applied machine learning scientist,
(3) include runnable code, showing readers *how* to solve problems in practice,
and (4) that allowed for rapid updates, both by us, and also by the community at large,
and (5) be complemented by a [forum](http://discuss.mxnet.io)
for interative discussion of technical details and to answer questions.

These goals were often in conflict.
Equations, theorems, and citations are best managed and laid out in LaTeX.
Code is best described in Python.
And webpages are native in HTML and JavaScript.
Furthermore, we want the content to be
accessible both as executable code, as a physical book,
as a downloadable PDF, and on the internet as a website.
At present there exist no tools and no workflow
perfectly suited to these demands, so we had to assemble our own.
We describe our approach in detail in the [appendix](../chapter_appendix/how-to-contribute.md).
We settled on Github to share the source and to allow for edits,
Jupyter notebooks for mixing code, equations and text,
Sphinx as a rendering engine to generate multiple outputs,
and Discourse for the forum.
While our system is not yet perfect, these choices provide a good compromise
among the competing concerns.
We believe that this might be the first book published using such an integrated workflow.

## Organization

Aside from a few preliminary notebooks that provide a crash course
in the basic mathematical background, each subsequent notebook introduces
both a reasonable number of new concepts and provides a single
self-contained working example, using a real dataset.
This presents an organizational challenge. Some models might logically
be grouped together in a single notebook.  And some ideas might be
best taught by executing several models in succession.  On the other
hand, there's a big advantage to adhering to a policy of *1 working
example, 1 notebook*: This makes it as easy as possible for you to
start your own research projects by plagiarising our code. Just copy a
single notebook and start modifying it.

We will interleave the runnable code with background material as needed.
In general, we will often err on the side of making tools
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

Many textbooks teach a series of topics, each in exhaustive detail.
For example, Chris Bishop's excellent textbook,
[Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738),
teaches each topic so thoroughly, that getting to the chapter
on linear regression requires a non-trivial amount of work.
While experts love this book precisely for its thoroughness,
for beginners, this property limits its usefulness as an introductory text.

In this book, we'll teach most concepts *just in time*.
In other words, you'll learn concepts at the very moment
that they are needed to accomplish some practical end.
While we take some time at the outset to teach
fundamental preliminaries, like linear algebra and probability.
we want you to taste the satisfaction of training your first model
before worrying about more exotic probability distributions.

If you're ready to get started, head over to [the introduction](../chapter_introduction/index.rst).
For everything else [open an issue on Github](https://github.com/diveintodeeplearning/d2l-en).

## Acknowledgments

We are indebted to the hundreds of contributors for both
the English draft and subsequently the Chinese version.
They helped improve the content and offered valuable feedback.
Their Github IDs and names (if provided) are: bowen0701, ChaiBapchya (Chaitanya Prakash Bapat), kirk86
, MLWhiz (Rahul Agarwal), muelleme, sundeepteki, vishaalkapoor, YaYaB.
Moreover, we thank Amazon Web Services for its generous support in writing this book.
Without the available time, resources, discussions with colleagues,
and continuous encouragement this book would not have happened.
