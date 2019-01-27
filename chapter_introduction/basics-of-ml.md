
## Basics of machine learning
When we considered the task of recognizing wake-words,
we put together a dataset consisting of snippets and labels.
We then described (albeit abstractly)
how you might train a machine learning model
to predict the label given a snippet.
This set-up, predicting labels from examples, is just one flavor of ML
and it's called *supervised learning*.
Even within deep learning, there are many other approaches,
and we'll discuss each in subsequent sections.
To get going with machine learning, we need four things:

1. Data
2. A model of how to transform the data
3. A loss function to measure how well we're doing
4. An algorithm to tweak the model parameters such that the loss function is minimized

### Data

Generally, the more data we have, the easier our job becomes.
When we have more data, we can train more powerful models. Data is at the heart of the resurgence of deep learning and many of most exciting models in deep learning don't work without large data sets. Here are some examples of the kinds of data machine learning practitioners often engage with:

* **Images:** Pictures taken by smartphones or harvested from the web, satellite images, photographs of medical conditions, ultrasounds, and radiologic images like CT scans and MRIs, etc.
* **Text:** Emails, high school essays, tweets, news articles, doctor's notes, books, and corpora of translated sentences, etc.
* **Audio:** Voice commands sent to smart devices like Amazon Echo, or iPhone or Android phones, audio books, phone calls, music recordings, etc.
* **Video:** Television programs and movies, YouTube videos, cell phone footage, home surveillance, multi-camera tracking, etc.
* **Structured data:** Webpages, electronic medical records, car rental records, electricity bills, etc.

### Models

Usually the data looks quite different from what we want to accomplish with it.
For example, we might have photos of people and want to know whether they appear to be happy.
We might desire a model capable of ingesting a high-resolution image and outputting a happiness score.
While some simple problems might be addressable with simple models, we're asking a lot in this case.
To do its job, our happiness detector needs to transform hundreds of thousands of low-level features (pixel values)
into something quite abstract on the other end (happiness scores).
Choosing the right model is hard, and different models are better suited to different datasets.
In this book, we'll be focusing mostly on deep neural networks.
These models consist of many successive transformations of the data that are chained together top to bottom,
thus the name *deep learning*.
On our way to discussing deep nets, we'll also discuss some simpler, shallower models.


###  Loss functions

To assess how well we're doing we need to compare the output from the model with the truth.
Loss functions give us a way of measuring how *bad* our output is.
For example, say we trained a model to infer a patient's heart rate from images.
If the model predicted that a patient's heart rate was 100bpm,
when the ground truth was actually 60bpm,
we need a way to communicate to the model that it's doing a lousy job.


Similarly if the model was assigning scores to emails indicating the probability that they are spam,
we'd need a way of telling the model when its predictions are bad.
Typically the *learning* part of machine learning consists of minimizing this loss function.
Usually, models have many parameters.
The best values of these parameters is what we need to 'learn', typically by minimizing the loss incurred on a *training data*
of observed data.
Unfortunately, doing well on the training data
doesn't guarantee that we will do well on (unseen) test data,
so we'll want to keep track of two quantities.

 * **Training Error:** This is the error on the dataset used to train our model by minimizing the loss on the training set. This is equivalent to doing well on all the practice exams that a student might use to prepare for the real exam. The results are encouraging, but by no means guarantee success on the final exam.
 * **Test Error:** This is the error incurred on an unseen test set. This can deviate quite a bit from the training error. This condition, when a model fails to generalize to unseen data, is called *overfitting*. In real-life terms, this is the equivalent of screwing up the real exam despite doing well on the practice exams.


### Optimization algorithms

Finally, to minimize the loss, we'll need some way of taking the model and its loss functions,
and searching for a set of parameters that minimizes the loss.
The most popular optimization algorithms for work on neural networks
follow an approach called gradient descent.
In short, they look to see, for each parameter which way the training set loss would move if you jiggled the parameter a little bit. They then update the parameter in the direction that reduces the loss.

In the following sections, we will discuss a few types of machine learning in some more detail. We begin with a list of *objectives*, i.e. a list of things that machine learning can do. Note that the objectives are complemented with a set of techniques of *how* to accomplish them, i.e. training, types of data, etc. The list below is really only sufficient to whet the readers' appetite and to give us a common language when we talk about problems. We will introduce a larger number of such problems as we go along.
