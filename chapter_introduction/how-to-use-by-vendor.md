# How to Use this Book

This book  provides a comprehensive introduction to all aspects of deep learning; from model construction to model training, as along with their applications in the fields of computer vision and natural language processing. Not only will the algorithm principles be explained, but we will also demonstrate their implementation and operation based on Apache MXNet.  Each section of the book is a Jupyter notebook which combines texts, formulas, images, code, and running results. Apart from reading them directly, you can also run each section for an interactive learning experience. 

## Target Reader

本书面向希望了解深度学习，特别是对实际使用深度学习感兴趣的大学生、工程师和研究人员。本书并不要求你有任何深度学习或者机器学习的背景知识，我们将从头开始解释每一个概念。虽然深度学习技术与应用的阐述涉及了数学和编程，但你只需了解基础的数学和编程，例如基础的线性代数、微分和概率，以及基础的Python编程。在附录中我们提供了本书所涉及的主要数学知识供你参考。如果你之前没有接触过Python，可以参考中文教程 http://www.runoob.com/python/python-tutorial.html 或英文教程 http://learnpython.org/ 。当然，如果你只对本书中的数学部分感兴趣，你可以忽略掉编程部分，反之亦然。


## Content and Structure

This book can be roughly divided into three sections:

* The first section (Chapters 1–3) covers the prerequisite work and basic knowledge required. Chapter 1 provides an introduction into the background of deep learning and covers how to properly use this book.  Chapter 2 provides the prerequisites required for hands-on deep learning practice, such as how to acquire and run the code covered in the book. Chapter 3 covers the most basic concepts and techniques of deep learning, like multi-layer perceptrons and model regularization. Reading just the first section will suffice if you just want to learn the most basic concepts and techniques of deep learning or are tight on time. 

* The second section (Chapters 4-6) focuses on modern deep learning techniques. Chapter 4 covers the various key components of deep learning calculations and lays the groundwork for the implementation of more complex models later on. Chapter 5 explains the convolutional neural networks that have made deep learning a success in the field of computer vision in the last few years. Chapter 6 covers the recurrent neural networks that have been commonly used over recent years to process sequence data. Reading through the second section will help you better grasp the concepts of modern deep learning techniques.

* The third section (Chapters 7-10) discusses computing performance and applications. Chapter 7 evaluates the various optimization algorithms used to train deep learning models. Chapter 8 examines several important factors that affect the deep learning computation performance. Chapters 9 and 10 respectively illustrate the major applications of deep learning in computer vision and natural language processing. This section can be considered optional, based on your interests. 

Figure 1.2 outlines the structure of the book.

![Book Structure. The arrows pointing from Chapter X to Chapter Y indicate that the knowledge of the former helps in the understanding of the latter’s content. If you want to quickly learn the basic concepts and techniques of deep learning, simply read through Chapters 1-3; if you want to master modern deep learning techniques, you will need to go a step further and continue reading through Chapters 4-6. Chapters 7-10 are optional, based on your interests. ](../img/book-org.svg)


## Code

Every section in this book is runnable.  The code can be edited and re-run to see how the results have been affected. We recognize the importance of an interactive learning experience when studying deep learning. Unfortunately, a sound theoretical explanation framework for deep learning has yet to be developed, and many arguments are cannot be explained in writing, but must be based simply on self-understanding. The text based explanation may be vague and might not be able to cover all the details.  To gain insight and further your understanding, you need to constantly change the code provided in this book, observing their running results and summarizing the entire process. 

The code in this book is based on the Apache MXNet framework.  MXNet is an open-source framework used in deep learning. It is the preferred choice of AWS (Amazon Cloud Services), and is used by many colleges and companies alike.  All code in this book have been tested successfully using MXNet 1.2.1.  However, due to the rapid development of deep learning, some of the code may not function correctly with later versions of MXNet. In event of such issues, please refer to the section ["Installation and Running](../chapter_prerequisite/install.md) to update the code and the runtime environment. Additionally, we encapsulate frequently-referred to functions, classes, etc. in this book in the `gluonbook` version 1.0.0 package to avoid repeated descriptions  The section in which these functions, classes, etc. are defined is listed in the appendix [“gluonbook package index”](../chapter_appendix/gluonbook.md)

This book can also serve as an MXNet primer. The main purpose of our code is to provide an alternative way of learning deep learning algorithms along with text, images, and formulas, and to add an interactive environment to aid in the understanding of the actual effects of individual models and algorithms on actual data. To familiarize yourself with the details of the implementation of deep learning algorithms, we only use the basic functionalities of MXNet's modules such as `ndarray`, `autograd`, `gluon`, etc. s. Even if you use other deep learning frameworks in your research or work, we hope that this code will help you better understand deep learning algorithms.


## Forum

本书的学习社区地址是 https://discuss.gluon.ai/ 。当你对书中某节内容有疑惑时，请扫一扫该节末尾的二维码参与该节内容的讨论。值得一提的是，在有关Kaggle比赛章节的讨论区中，众多社区成员提供了丰富的高水平方法。我们强烈推荐大家积极参与学习社区中的讨论，并相信你一定会有所收获。本书作者和MXNet开发人员也时常参与社区中的讨论。

## Summary

* MXNet has been selected as the deep learning framework used in this book.
* This book seeks to create a multi-faceted, interactive deep learning experience 


## exercise

* 在本书的学习社区 https://discuss.gluon.ai/ 上注册一个账号。搜索关键字Kaggle，浏览其中回复量最大的几个帖子。


## Scan the QR code to access the [forum](https://discuss.gluon.ai/t/topic/6915)

![](../img/qr_how-to-use.svg)
