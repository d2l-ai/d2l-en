# Preface

Just a few years ago, there were no legions of deep learning scientists
developing intelligent products and services at major companies and startups.
When we entered the field, machine learning 
did not command headlines in daily newspapers.
Our parents had no idea what machine learning was,
let alone why we might prefer it to a career in medicine or law.
Machine learning was a blue skies academic discipline
whose industrial significance was limited
to a narrow set of real-world applications,
including speech recognition and computer vision.
Moreover, many of these applications
required so much domain knowledge
that they were often regarded as entirely separate areas 
for which machine learning was one small component.
At that time, neural networks---the 
predecessors of the deep learning methods
that we focus on in this book---were 
generally regarded as outmoded.


In just the past few years, deep learning has taken the world by surprise,
driving rapid progress in such diverse fields 
as computer vision, natural language processing, 
automatic speech recognition, reinforcement learning, 
and biomedical informatics.
Moreover, the success of deep learning
on so many tasks of practical interest
has even catalyzed developments 
in theoretical machine learning and statistics.
With these advances in hand, 
we can now build cars that drive themselves
with more autonomy than ever before 
(and less autonomy than some companies might have you believe),
smart reply systems that automatically draft the most mundane emails,
helping people dig out from oppressively large inboxes,
and software agents that dominate the world's best humans
at board games like Go, a feat once thought to be decades away.
Already, these tools exert ever-wider impacts on industry and society,
changing the way movies are made, diseases are diagnosed,
and playing a growing role in basic sciences---from astrophysics to biology.



## About This Book

This book represents our attempt to make deep learning approachable,
teaching you the *concepts*, the *context*, and the *code*.

### One Medium Combining Code, Math, and HTML

For any computing technology to reach its full impact,
it must be well-understood, well-documented, and supported by
mature, well-maintained tools.
The key ideas should be clearly distilled,
minimizing the onboarding time needing 
to bring new practitioners up to date.
Mature libraries should automate common tasks,
and exemplar code should make it easy for practitioners
to modify, apply, and extend common applications to suit their needs.
Take dynamic web applications as an example.
Despite a large number of companies, like Amazon,
developing successful database-driven web applications in the 1990s,
the potential of this technology to aid creative entrepreneurs
has been realized to a far greater degree in the past ten years,
owing in part to the development of powerful, well-documented frameworks.


Testing the potential of deep learning presents unique challenges
because any single application brings together various disciplines.
Applying deep learning requires simultaneously understanding
(i) the motivations for casting a problem in a particular way;
(ii) the mathematical form of a given model;
(iii) the optimization algorithms for fitting the models to data;
(iv) the statistical principles that tell us 
when we should expect our models 
to generalize to unseen data
and practical methods for certifying 
that they have, in fact, generalized;
and (v) the engineering techniques
required to train models efficiently,
navigating the pitfalls of numerical computing
and getting the most out of available hardware.
Teaching both the critical thinking skills 
required to formulate problems,
the mathematics to solve them,
and the software tools to implement those solutions 
all in one place presents formidable challenges.
Our goal in this book is to present a unified resource
to bring would-be practitioners up to speed.

When we started this book project,
there were no resources that simultaneously 
(i) remained up to date;
(ii) covered the breadth of modern machine learning practices 
with sufficient technical depth;
and (iii) interleaved exposition of 
the quality one expects of a textbook 
with the clean runnable code
that one expects of a hands-on tutorial.
We found plenty of code examples for
how to use a given deep learning framework
(e.g., how to do basic numerical computing with matrices in TensorFlow)
or for implementing particular techniques
(e.g., code snippets for LeNet, AlexNet, ResNet, etc.)
scattered across various blog posts and GitHub repositories.
However, these examples typically focused on
*how* to implement a given approach,
but left out the discussion of 
*why* certain algorithmic decisions are made.
While some interactive resources 
have popped up sporadically
to address a particular topic, 
e.g., the engaging blog posts
published on the website [Distill](http://distill.pub), or personal blogs,
they only covered selected topics in deep learning,
and often lacked associated code.
On the other hand, while several deep learning textbooks 
have emerged---e.g., :cite:`Goodfellow.Bengio.Courville.2016`, 
which offers a comprehensive survey 
on the basics of deep learning---these 
resources do not marry the descriptions
to realizations of the concepts in code,
sometimes leaving readers clueless 
as to how to implement them.
Moreover, too many resources 
are hidden behind the paywalls
of commercial course providers.

We set out to create a resource that could
(i) be freely available for everyone;
(ii) offer sufficient technical depth 
to provide a starting point on the path
to actually becoming an applied machine learning scientist;
(iii) include runnable code, showing readers
*how* to solve problems in practice;
(iv) allow for rapid updates, both by us
and also by the community at large;
and (v) be complemented by a [forum](http://discuss.d2l.ai)
for interactive discussion of technical details and to answer questions.

These goals were often in conflict.
Equations, theorems, and citations 
are best managed and laid out in LaTeX.
Code is best described in Python.
And webpages are native in HTML and JavaScript.
Furthermore, we want the content to be
accessible both as executable code, as a physical book,
as a downloadable PDF, and on the Internet as a website.
No workflows seemed suited to these demands, 
so we decided to assemble our own (:numref:`sec_how_to_contribute`).
We settled on GitHub to share the source 
and to facilitate community contributions;
Jupyter notebooks for mixing code, equations and text;
Sphinx as a rendering engine; 
and Discourse as a discussion platform.
While our system is not perfect,
these choices strike a compromise 
among the competing concerns.
We believe that *Dive into Deep Learning*
might be the first book published
using such an integrated workflow.


### Learning by Doing

Many textbooks present concepts in succession, 
covering each in exhaustive detail.
For example, Chris Bishop's excellent textbook :cite:`Bishop.2006`,
teaches each topic so thoroughly
that getting to the chapter
on linear regression requires 
a non-trivial amount of work.
While experts love this book 
precisely for its thoroughness,
for true beginners, this property limits 
its usefulness as an introductory text.

In this book, we teach most concepts *just in time*.
In other words, you will learn concepts at the very moment
that they are needed to accomplish some practical end.
While we take some time at the outset to teach
fundamental preliminaries, like linear algebra and probability,
we want you to taste the satisfaction of training your first model
before worrying about more esoteric concepts.

Aside from a few preliminary notebooks that provide a crash course
in the basic mathematical background,
each subsequent chapter introduces both a reasonable number of new concepts
and provides several self-contained working examples, using real datasets.
This presented an organizational challenge.
Some models might logically be grouped together in a single notebook.
And some ideas might be best taught 
by executing several models in succession.
On the other hand, there is a big advantage to adhering
to a policy of *one working example, one notebook*:
This makes it as easy as possible for you to
start your own research projects by leveraging our code.
Just copy a notebook and start modifying it.

Throughout, we interleave the runnable code
with background material as needed.
In general, we err on the side of making tools
available before explaining them fully
(often filling in the background later).
For instance, we might use *stochastic gradient descent*
before explaining why it is useful 
or offering intuitions for why it works.
This helps to give practitioners the necessary
ammunition to solve problems quickly,
at the expense of requiring the reader
to trust us with some curatorial decisions.

This book teaches deep learning concepts from scratch.
Sometimes, we delve into fine details about models
that would typically be hidden from users
by modern deep learning frameworks.
This comes up especially in the basic tutorials,
where we want you to understand everything
that happens in a given layer or optimizer.
In these cases, we often present 
two versions of the example:
one where we implement everything from scratch,
relying only on NumPy-like functionality
and automatic differentiation,
and a more practical example,
where we write succinct code 
using the high-level APIs of deep learning frameworks.
After explaining how some component works,
we rely on the high-level API in subsequent tutorials.


### Content and Structure

The book can be divided into roughly three parts,
focusing on preliminaries, 
deep learning techniques,
and advanced topics
focused on real systems
and applications (:numref:`fig_book_org`).

![Book structure](../img/book-org.svg)
:label:`fig_book_org`


* **Part 1: Basics and Preliminaries.**
:numref:`chap_introduction` offers 
an introduction to deep learning.
Then, in :numref:`chap_preliminaries`,
we quickly bring you up to speed 
on the prerequisites required
for hands-on deep learning, 
such as how to store and manipulate data,
and how to apply various numerical operations 
based on basic concepts from linear algebra, 
calculus, and probability.
:numref:`chap_regression` and :numref:`chap_perceptrons`
cover the most basic concepts and techniques in deep learning,
including regression and classification;
linear models; multilayer perceptrons;
and overfitting and regularization.

* **Part 2: Modern Deep Learning Techniques.**
:numref:`chap_computation` describes
the key computational components
of deep learning systems
and lays the groundwork
for our subsequent implementations
of more complex models.
Next, :numref:`chap_cnn` and :numref:`chap_modern_cnn`
introduce convolutional neural networks (CNNs), 
powerful tools that form the backbone 
of most modern computer vision systems.
Similarly, :numref:`chap_rnn` and :numref:`chap_modern_rnn`
introduce recurrent neural networks (RNNs), 
models that exploit sequential (e.g., temporal) 
structure in data and are commonly used
for natural language processing 
and time series prediction.
In :numref:`chap_attention-and-transformers`, 
we introduce a relatively new class of models
based on so-called *attention mechanisms*
that has displaced RNNs as the dominant architecture
for most natural language processing tasks.
These sections will bring you up to speed 
on the most powerful and general tools
that are widely used by deep learning practitioners.

* **Part 3: Scalability, Efficiency, and Applications.**
In :numref:`chap_optimization`,
we discuss several common optimization algorithms
used to train deep learning models.
Next, in :numref:`chap_performance`,
we examine several key factors
that influence the computational performance 
of deep learning code.
Then, in :numref:`chap_cv`,
we illustrate major applications 
of deep learning in computer vision.
Finally, in :numref:`chap_nlp_pretrain` and :numref:`chap_nlp_app`,
we demonstrate how to pretrain language representation models 
and apply them to natural language processing tasks.
This part is available [online](https://d2l.ai).


### Code
:label:`sec_code`

Most sections of this book feature executable code.
We believe that some intuitions are best developed
via trial and error,
tweaking the code in small ways and observing the results.
Ideally, an elegant mathematical theory might tell us
precisely how to tweak our code to achieve a desired result.
However, deep learning practitioners today
must often tread where no solid theory provides guidance. 
Despite our best attempts, formal explanations 
for the efficacy of various techniques are still lacking,
both because the mathematics to characterize these models
can be so difficult,
because the explanation likely depends on properties 
of the data that currently lack clear definitions,
and because serious inquiry on these topics
has just recently kicked into high gear.
We are hopeful that as the theory of deep learning progresses,
each future edition of this book will provide insights 
that eclipse those presently available.

To avoid unnecessary repetition, we encapsulate
some of our most frequently imported and used
functions and classes in the `d2l` package.
Throughout, we mark blocks of code
(such as functions, classes,
or collection of import statements) with `#@save`
to indicate that they will be accessed later
via the `d2l` package.
We offer a detailed overview 
of these functions and classes in :numref:`sec_d2l`.
The `d2l` package is lightweight and only requires
the following dependencies:

```{.python .input}
#@tab all
#@save
import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
Most of the code in this book is based on Apache MXNet,
an open-source framework for deep learning
that is the preferred choice 
of AWS (Amazon Web Services),
as well as many colleges and companies.
All of the code in this book has passed tests 
under the newest MXNet version.
However, due to the rapid development of deep learning, 
some code *in the print edition* 
may not work properly in future versions of MXNet.
We plan to keep the online version up-to-date.
In case you encounter any problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from MXNet.
:end_tab:

:begin_tab:`pytorch`
Most of the code in this book is based on PyTorch,
an extremely popular open-source framework
that has been enthusiastically embraced 
by the deep learning research community.
All of the code in this book has passed tests 
under the latest stable version of PyTorch.
However, due to the rapid development of deep learning,
some code *in the print edition* 
may not work properly in future versions of PyTorch.
We plan to keep the online version up-to-date.
In case you encounter any problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from PyTorch.
:end_tab:

:begin_tab:`tensorflow`
Most of the code in this book is based on TensorFlow,
an open-source framework for deep learning
that is widely adopted in industry
and popular among researchers.
All of the code in this book has passed tests 
under the latest stable version TensorFlow.
However, due to the rapid development of deep learning, 
some code *in the print edition* 
may not work properly in future versions of TensorFlow.
We plan to keep the online version up-to-date.
In case you encounter any problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from TensorFlow.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from PIL import Image
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

### Target Audience

This book is for students (undergraduate or graduate),
engineers, and researchers, who seek a solid grasp
of the practical techniques of deep learning.
Because we explain every concept from scratch,
no previous background in deep learning or machine learning is required.
Fully explaining the methods of deep learning
requires some mathematics and programming,
but we will only assume that you come in with some basics,
including modest amounts of linear algebra, 
calculus, probability, and Python programming.
Just in case you forget the basics,
the Appendix provides a refresher 
on most of the mathematics 
you will find in this book.
Most of the time, we will prioritize 
intuition and ideas
over mathematical rigor.
If you would like to extend these foundations 
beyond the prerequisites to understand our book,
we happily recommend some other terrific resources:
Linear Analysis by Bela Bollobas :cite:`Bollobas.1999`
covers linear algebra and functional analysis in great depth.
All of Statistics :cite:`Wasserman.2013` 
provides a marvelous introduction to statistics.
Joe Blitzstein's [books](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918) 
and [courses](https://projects.iq.harvard.edu/stat110/home) 
on probability and inference are pedagogical gems.
And if you have not used Python before,
you may want to peruse this [Python tutorial](http://learnpython.org/).


### Forum

Associated with this book, we have launched a discussion forum,
located at [discuss.d2l.ai](https://discuss.d2l.ai/).
When you have questions on any section of the book,
you can find a link to the associated discussion page
at the end of each notebook.


## Acknowledgments

We are indebted to the hundreds of contributors for both
the English and the Chinese drafts.
They helped improve the content and offered valuable feedback.
Specifically, we thank every contributor of this English draft
for making it better for everyone.
Their GitHub IDs or names are (in no particular order):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan,
atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore,
Joji Joseph, Anthony Biel, Zeming Zhao.

We thank Amazon Web Services, especially Swami Sivasubramanian, Peter DeSantis, Adam Selipsky,
and Andrew Jassy for their generous support in writing this book. 
Without the available time, resources, discussions with colleagues, 
and continuous encouragement, this book would not have happened.


## Summary

Deep learning has revolutionized pattern recognition, 
introducing technology that now powers a wide range of  technologies, 
in such diverse fields as computer vision,
natural language processing,
and automatic speech recognition.
To successfully apply deep learning, 
you must understand how to cast a problem,
the basic mathematics of modeling,
the algorithms for fitting your models to data,
and the engineering techniques to implement it all.
This book presents a comprehensive resource, 
including prose, figures, mathematics, and code, all in one place.
To ask (or answer) questions related to this book,
visit our forum at https://discuss.d2l.ai/.
All of our notebooks are available for download
on the [D2L.ai website](https://d2l.ai)
and on [GitHub](https://github.com/d2l-ai/d2l-en).


## Exercises

1. Register an account on the discussion forum of this book [discuss.d2l.ai](https://discuss.d2l.ai/).
1. Install Python on your computer.
1. Follow the links at the bottom of the section to the forum, where you will be able to seek out help and discuss the book and find answers to your questions by engaging the authors and broader community.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/186)
:end_tab:
