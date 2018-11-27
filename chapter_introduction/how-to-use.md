# Using this Book

We aim to provide a comprehensive introduction to all aspects of deep learning from model construction to model training, as well as  applications in computer vision and natural language processing. We will not only explain the principles of the algorithms, but also demonstrate their implementation and operation in Apache MXNet. Each section of the book is a Jupyter notebook,  combining text, formulae, images, code, and running results. Not only can you read them directly, but you can run them to get an interactive learning experience. But since it is an introduction, we can only cover things so far. It is up to you, the reader, to explore further, to play with the toolboxes, compiler, and examples, tutorials and code snippets that are available in the research community. Enjoy the journey!

## Target Audience

This book is for college students, engineers, and researchers who wish to learn deep learning, especially for those who are interested in applying deep learning in practice. Readers need not have a background in deep learning or machine learning. We will explain every concept from scratch. Although illustrations of deep learning techniques and applications involve mathematics and programming, you only need to know their basics, such as basic linear algebra, calculus, and probability, and basic Python programming. In the appendix we provide most of the mathematics covered in this book for your reference. Since it's an introduction, we prioritize intuition and ideas over mathematical rigor. There are many terrific books which can lead the interested reader further. For instance [Linear Analysis](https://www.amazon.com/Linear-Analysis-Introductory-Cambridge-Mathematical/dp/0521655773) by Bela Bollobas covers linear algebra and functional analysis in great depth. [All of Statistics](https://www.amazon.com/All-Statistics-Statistical-Inference-Springer/dp/0387402721) is a terrific guide to statistics. And if you have not used Python before, you may want to peruse the [Python tutorial](http://learnpython.org/). Of course, if you are only interested in the mathematical part, you can ignore the programming part, and vice versa.


## Content and Structure

The book can be roughly divided into three sections:

* The first part covers prerequisites and basics. The first chapter offers an [Introduction to Deep Learning](../chapter_introduction/index.md) and how to use this book. [A Taste of Deep Learning](../chapter_crashcourse/index.md) provides the prerequisites required for hands-on deep learning, such as how to acquire and run the codes covered in the book.  [Deep Learning Basics](../chapter_deep-learning-basics/index.md) covers the most basic concepts and techniques of deep learning, such as multi-layer perceptrons and regularization. If you are short on time or you only want to learn only about the most basic concepts and techniques of deep learning, it is sufficient to read the first section only.

* The next three chapters focus on modern deep learning techniques. [Deep Learning Computation](../chapter_deep-learning-computation/index.md) describes the various key components of deep learning calculations and lays the groundwork for the later implementation of more complex models. [Convolutional Neural Networks](../chapter_convolutional-neural-networks/index.md) are explained next. They have made deep learning a success in computer vision in recent years. [Recurrent Neural Networks](../chapter_recurrent-neural-networks/index.md) are commonly used to process sequence data in recent years. Reading through the second section will help you grasp modern deep learning techniques.

* Part three discusses scalability, efficiency and applications. In particular, we discuss various [Optimization Algorithms](../chapter_optimization/index.md) used to train deep learning models. The next chapter examines several important factors that affect the [Performance](../chapter_computational-performance/index.md) of deep learning computation, such as regularization. Chapters 9 and 10  illustrate major applications of deep learning in computer vision and natural language processing respectively. This part is optional, depending on the reader's interests.

An outline of the book is given below. The arrows provide a graph of prerequisites. If you want to learn the basic concepts and techniques of deep learning in a short time, simply read through the first three chapters; if you want to advance further, you will need the next three chapters. The last four chapters are optional, based on the reader's interests.

![Book structure](../img/book-org.svg)


## Code

This book features executable code in every section. The codes can be modified and re-run to see how it affects the results. We recognize the importance of an interactive learning experience in deep learning. Unfortunately, deep learning remains to be poorly understood in theoretical terms. As a result, many arguments rely heavily on phenomenological experience that is best gained by experimentation with the codes provided. The textual explanation may be insufficient to cover all the details, despite our best attempts. We are hopeful that this situation will improve in the future, as more theoretical progress is made. For now, we strongly advise that the reader further his understanding and gain insight by changing the codes, observing their outcomes and summarizing the whole process.

Codes in this book are based on the Apache MXNet. MXNet is an open-source framework for deep learning  which is the preferred choice of AWS (Amazon Cloud Services). It is used in many colleges and companies. All the codes in this book have passed the test under MXNet 1.2.1. However, due to the rapid development of deep learning, some of the codes *in the print edition* may not work properly in future versions of MXNet. The online version will remain up-to-date, though. In case of such problems, please refer to the section ["Installation and Running](../chapter_prerequisite/install.md) to update the codes and their runtime environment. In addition, to avoid unnecessary repetition, we encapsulate the frequently-imported and referred-to functions, classes, etc. in this book in the `gluonbook` package with version number 1.0.0.  We give a detailed overview of these functions and classes in the appendix [“gluonbook package index”](../chapter_appendix/gluonbook.md)

This book can also serve as an MXNet primer. The main purpose of our codes is to provide another way to learn deep learning algorithms in addition to text, images, and formulae. This book offers an interactive environment to understand the actual effects of individual models and algorithms on actual data. We only use the basic functionalities of MXNet's modules such as `ndarray`, `autograd`, `gluon`, etc. to familiarize yourself with the implementation details of deep learning algorithms. Even if you use other deep learning frameworks in your research or work, we hope that these codes will help you better understand deep learning algorithms.

## Forum

The discussion forum of this book is [discuss.mxnet.io](https://discuss.mxnet.io/). When you have questions on any section of the book, please scan the QR code at the end of the section to participate in its discussions. The authors of this book and MXNet developers are frequent visitors and participants on the forum.

## Problems

1. Register an account on the discussion forum of this book [discuss.mxnet.io](https://discuss.mxnet.io/).
1. Install Python on your computer.

<div id="discuss" topic_id="2311"></div>
