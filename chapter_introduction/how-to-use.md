# How to Use This Book

We aim to provide a comprehensive introduction to all aspects of deep learning from model construction to model training, as well as  applications in computer vision and natural language processing. We will not only explain the principles of the algorithms, but also demonstrate their implementation and operation in Apache MXNet. Each section of the book is a Jupyter notebook,  combining text, formulae, images, code, and running results. Not only can you read them directly, but you can run them to get an interactive learning experience.

## Target Audience

This book is for college students, engineers, and researchers who wish to learn deep learning, especially for those who are interested in applying deep learning in practice. Readers need not have a background in deep learning or machine learning. We will explain every concept from scratch. Although illustrations of deep learning techniques and applications involve mathematics and programming, you only need to know their basics, such as basic linear algebra, calculus, and probability, and basic Python programming. In the appendix we provide most of the mathematics covered in this book for your reference. If you have not used Python before, you may want to peruse the [Python tutorial](http://learnpython.org/). Of course, if you are only interested in the mathematical part, you can ignore the programming part, and vice versa.


## Content and Structure

The book can be roughly divided into three sections:

* The first part (Chapters 1–3) covers prerequisites and basics. Chapter 1 gives a background on deep learning and how to use this book. Chapter 2 provides the prerequisites required for hands-on deep learning, such as how to acquire and run the codes covered in the book. Chapter 3 covers the most basic concepts and techniques of deep learning, such as multi-layer perceptrons and regularization. If you are short on time or you only want to learn only about the most basic concepts and techniques of deep learning, it is sufficient to read the first section only.

* The second part (Chapters 4-6) focuses on modern deep learning techniques. Chapter 4 describes the various key components of deep learning calculations and lays the groundwork for the later implementation of more complex models. Chapter 5 explains the convolutional neural networks that have made deep learning a success in computer vision in recent years. Chapter 6 describes the recurrent neural networks that are commonly used to process sequence data in recent years. Reading through the second section will help you grasp modern deep learning techniques.

* Part three (Chapters 7-10) discusses scalability, efficiency and applications. Chapter 7 introduces various optimization algorithms used to train deep learning models. Chapter 8 examines several important factors that affect the performance of deep learning computation, such as regularization. Chapters 9 and 10  illustrate major applications of deep learning in computer vision and natural language processing respectively. This part is optional, depending on the reader's interests.

Figure 1.2 outlines the structure of the book.

![Book structure. The arrows provide a graph of prerequisites. If you want to learn the basic concepts and techniques of deep learning in a short time, simply read through Chapters 1-3; if you want to master modern deep learning techniques, you need to go a step further to read Chapters 4-6. Chapters 7-10 are optional, based on the reader's interests.](../img/book-org.svg)


## Code

This book features executable code in every section. The codes can be modified and re-run to see how it affects the results. We recognize the importance of an interactive learning experience in deep learning. Unfortunately, deep learning remains to be poorly understood in theoretical terms. As a result, many arguments rely heavily on phenomenological experience that is best gained by experimentation with the codes provided. The textual explanation may be insufficient to cover all the details, despite our best attempts. We are hopeful that this situation will improve in the future, as more theoretical progress is made. For now, we strongly advise that the reader further his understanding and gain insight by changing the codes, observing their outcomes and summarizing the whole process.

Codes in this book are based on the Apache MXNet. MXNet is an open-source framework for deep learning  which is the preferred choice of AWS (Amazon Cloud Services). It is used in many colleges and companies. All the codes in this book have passed the test under MXNet 1.2.1. However, due to the rapid development of deep learning, some of the codes *in the print edition* may not work properly in future versions of MXNet. The online version will remain up-to-date, though. In case of such problems, please refer to the section ["Installation and Running](../chapter_prerequisite/install.md) to update the codes and their runtime environment. In addition, to avoid unnecessary repetition, we encapsulate the frequently-imported and referred-to functions, classes, etc. in this book in the `gluonbook` package with version number 1.0.0.  We give a detailed overview of these functions and classes in the appendix [“gluonbook package index”](../chapter_appendix/gluonbook.md)

This book can also serve as an MXNet primer. The main purpose of our codes is to provide another way to learn deep learning algorithms in addition to text, images, and formulae. This book offers an interactive environment to understand the actual effects of individual models and algorithms on actual data. We only use the basic functionalities of MXNet's modules such as `ndarray`, `autograd`, `gluon`, etc. to familiarize yourself with the implementation details of deep learning algorithms. Even if you use other deep learning frameworks in your research or work, we hope that these codes will help you better understand deep learning algorithms.

## Forum

The discussion forum of this book is https://discuss.mxnet.io/ . When you have questions on any section of the book, please scan the QR code at the end of the section to participate in its discussions. The authors of this book and MXNet developers are frequent visitors and participants on the forum.

## Exercise

* Register an account on the discussion forum of this book https://discuss.mxnet.io/ .


## Scan the QR code to get to the [forum](https://discuss.mxnet.io/t/how-to-use-this-book-discussions/2011)

![](../img/qr_how-to-use.svg)
