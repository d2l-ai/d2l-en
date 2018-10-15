# How to use this book

This book will provide a comprehensive introduction to all aspects of deep learning from model construction to model training, as well as their applications in computer vision and natural language processing. We will not only explain the principles of the algorithm, but also demonstrate their implementation and operation based on Apache MXNet. Each section of the book is a Jupyter notebook,  which combines texts, formulas, images, codes, and running results. Not only can you read them directly, but you can run them to get an interactive learning experience.

## Target reader

This book is for college students, engineers, and researchers who wish to learn deep learning, especially for those who are interested in applying deep learning into practice. This book does not require that you have any background in deep learning or machine learning. We will explain every concept from scratch. Although illustrations of deep learning techniques and applications involve mathematics and programming, you only need to know their basics, such as basic linear algebra, calculus, and probability, and basic Python programming. In appendix we provide most of the mathematics covered in this book for your reference. If you have not used Python before, you may refer to the tutorial http://learnpython.org/ . Of course, if you are only interested in the mathematical part, you can ignore the programming part, and vice versa.


## Content and structure

The book can be roughly divided into three sections:

* The first section (Chapters 1–3) covers prerequisite work and basic knowledge concerned. Chapter 1 introduces the background of deep learning and how to use this book. Chapter 2 provides the prerequisites required for hands-on deep learning, such as how to acquire and run the codes covered in the book. Chapter 3 covers the most basic concepts and techniques of deep learning, such as multi-layer perceptrons and model regularization. If time does not permit or you only want to learn about the most basic concepts and techniques of deep learning, then it suffices to read the first section only.

* The second section (Chapters 4-6) focuses on modern deep learning techniques. Chapter 4 describes the various key components of deep learning calculations and lays the groundwork for the later implementation of more complex models. Chapter 5 explains the convolutional neural networks that have made deep learning a success in computer vision in recent years. Chapter 6 describes the recurrent neural networks that are commonly used to process sequence data in recent years. Reading through the second section will help you grasp modern deep learning techniques.

* The third section (Chapters 7-10) discusses computing performance and applications. Chapter 7 evaluates various optimization algorithms used to train deep learning models. Chapter 8 examines several important factors that affect the performance of deep learning computation. Chapters 9 and 10 respectively illustrate the major applications of deep learning in computer vision and natural language processing. This section is for you to optionally read based on your interests.

Figure 1.2 outlines the structure of the book.

![Book structure. The arrows pointing from Chapter X to Chapter Y indicate that the knowledge of the former helps to understand the content of the latter. If you want to learn the basic concepts and techniques of deep learning in a short time, simply read through Chapters 1-3; if you want to master modern deep learning techniques, you need to go a step further to read Chapters 4-6. Chapters 7-10 are for you to optionally read based on your interests. ](../img/book-org.svg)


## Code

This book features runnability of every section. The codes can be changed and re-run to see how it affects the results. We recognize the important role of interactive learning experience in deep learning. But, unfortunately, there is still no sound theoretical explanation framework for deep learning, and many arguments are not explainable in words, but simply based on self-understanding. The textual explanation may be pale and not enough to cover all the details. You need to further your understanding and gain insight by constantly changing the codes, observing their running results and summarizing the whole process.

Codes in this book are based on the Apache MXNet. MXNet is an open-source framework for deep learning  which is the preferred choice of AWS (Amazon Cloud Services), and is used in many colleges and companies. All the codes in this book have passed the test under MXNet 1.2.1. However, due to the rapid development of deep learning, some of the codes may not work properly in future versions of MXNet. In event of such problems, please refer to the section ["Installation and Running](../chapter_prerequisite/install.md) to update the codes and their runtime environment. In addition, to avoid repeated descriptions, we encapsulate the frequently-referred functions, classes, etc. in this book in the `gluonbook` package with version number 1.0.0.  The section in which these functions and classes, etc. are defined is listed in the appendix [“gluonbook package index”](../chapter_appendix/gluonbook.md)

This book can serve as an MXNet primer. The main purpose of our codes is to provide another way to learn deep learning algorithms in addition to text, images, and formulas, and an interactive environment to understand the actual effects of individual models and algorithms on actual data. We only use the basic functionalities of MXNet's modules such as `ndarray`, `autograd`, `gluon`, etc. to familiarize yourself with the implementation details of deep learning algorithms. Even if you use other deep learning frameworks in your research or work, we hope that these codes will help you better understand deep learning algorithms.


## Forum

本书的学习社区地址是 https://discuss.gluon.ai/ 。当你对书中某节内容有疑惑时，请扫一扫该节末尾的二维码参与该节内容的讨论。值得一提的是，在有关Kaggle比赛章节的讨论区中，众多社区成员提供了丰富的高水平方法。我们强烈推荐大家积极参与学习社区中的讨论，并相信你一定会有所收获。本书作者和MXNet开发人员也时常参与社区中的讨论。

## Summary

* MXNet is selected as the framework for deep learning in this book.
* This book seeks to create a multi-faceted interactive learning experience with deep learning.


## exercise

* 在本书的学习社区 https://discuss.gluon.ai/ 上注册一个账号。搜索关键字Kaggle，浏览其中回复量最大的几个帖子。


## Scan the QR code to get to the [forum](https://discuss.gluon.ai/t/topic/6915)

![](../img/qr_how-to-use.svg)
