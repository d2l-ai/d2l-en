
# Introduction

Before we could begin writing,
the authors of this book, 
like much of the work force, 
had to become caffeinated. 
We hopped in the car and started driving.
Having an Android, Alex called out "Okay Google",
awakening the phone's voice recognition system.
Then Mu commanded "directions to Blue Bottle coffee shop".
The phone quickly displayed the transcription of his command.
It also recognized that we were asking for directions 
and launched the Maps application to fulfill our request. 
Once launched, the Maps app identified a number of routes. 
Next to each route, the phone displayed a predicted transit time.
While we fabricated this story for pedagogical convenience,
it demonstrates that in the span of just a few seconds, 
our everyday interactions with a smartphone
can engage several machine learning models.


If you've never worked with machine learning before,
you might be wondering what the hell we're talking about. 
You might ask, "isn't that just programming?"
or "what does *machine learning* even mean?"
First, to be clear, we implement all machine learning algorithms 
by writing computer programs. 
Indeed, we use the same languages and hardware 
as other fields of computer science,
but not all computer programs involve machine learning. 
In response to the second question, 
precisely defining a field of study 
as vast as machine learning is hard. 
It's a bit like answering, "what is math?". 
But we'll try to give you enough intuition to get started.


## A motivating example 

Most of the computer programs we interact with every day 
can be coded up from first principles.
When you add an item to your shopping cart, 
you trigger an e-commerce application to store an entry 
in a *shopping cart* database table, 
associating your user ID with the product's ID. 
We can write such a program from first principles, 
launch without ever having seen a real customer.
When it's this easy to write an application 
*you should not be using machine learning*. 

Fortunately (for the community of ML scientists), however, 
for many problems, solutions aren't so easy.
Returning to our fake story about going to get coffee,
imagine just writing a program to respond to a *wake word* 
like "Alexa", "Okay, Google" or "Siri".
Try coding it up in a room by yourself 
with nothing but a computer and a code editor. 
How would you write such a program from first principles?
Think about it... the problem is hard.
Every second, the microphone will collect roughly 44,000 samples.
What rule could map reliably from a snippet of raw audio 
to confident predictions ``{yes, no}`` 
on whether the snippet contains the wake word?
If you're stuck, don't worry. 
We don't know how to write such a program from scratch either. 
That's why we use machine learning. 

![](../img/wake-word.png)

Here's the trick. 
Often, even when we don't know how to tell a computer 
explicitly how to map from inputs to outputs,
we are nonetheless capable of performing the cognitive feat ourselves.
In other words, even if you don't know *how to program a computer* to recognize the word "Alexa",
you yourself *are able* to recognize the word "Alexa". 
Armed with this ability, 
we can collect a huge *data set* containing examples of audio
and label those that *do* and that *do not* contain the wake word.
In the machine learning approach, we do not design a system *explicitly* to recognize 
wake words right away. 
Instead, we define a flexible program with a number of *parameters*. 
These are knobs that we can tune to change the behavior of the program.
We call this program a model.
Generally, our model is just a machine that transforms its input into some output.
In this case, the model receives as *input* a snippet of audio,
and it generates as output an answer ``{yes, no}``,
which we hope reflects whether (or not) the snippet contains the wake word.

If we choose the right kind of model, 
then there should exist one setting of the knobs
such that the model fires ``yes`` every time it hears the word "Alexa".
There should also be another setting of the knobs that might fire ``yes`` 
on the word "Apricot". 
We expect that the same model should apply to "Alexa" recognition and "Apricot" recognition
because these are similar tasks. 
However, we might need a different model to deal with fundamentally different inputs or outputs. 
For example, we might choose a different sort of machine to map from images to captions,
or from English sentences to Chinese sentences. 

As you might guess, if we just set the knobs randomly,
the model will probably recognize neither "Alexa", "Apricot",
nor any other English word. 
Generally, in deep learning, the *learning*
refers precisely 
to updating the model's behavior (by twisting the knobs)
over the course of a *training period*. 

The training process usually looks like this:

1. Start off with a randomly initialized model that can't do anything useful.
1. Grab some of your labeled data (e.g. audio snippets and corresponding ``{yes,no}`` labels)
1. Tweak the knobs so the model sucks less with respect to those examples
1. Repeat until the model is awesome.

![](../img/ml-loop.png)

To summarize, rather than code up a wake word recognizer, 
we code up a program that can *learn* to recognize wake words, 
*if we present it with a large labeled dataset*.
You can think of this act
of determining a program's behavior by presenting it with a dataset
as *programming with data*.

We can 'program' a cat detector by providing our machine learning system with many examples of cats and dogs, such as the images below:

|![](../img/cat1.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|
|:---------------:|:---------------:|:---------------:|:---------------:|
|cat|cat|dog|dog|

This way the detector will eventually learn to emit a very large positive number if it's a cat, a very large negative number if it's a dog, and something closer to zero if it isn't sure, but this is just barely scratching the surface of what machine learning can do.


## The dizzying versatility of machine learning

This is the core idea behind machine learning:
Rather than code programs with fixed behavior,
we design programs with the ability to improve
as they acquire more experience. 
This basic idea can take many forms.
Machine learning can address many different application domains, 
involve many different types of models,
and update them according to many different learning algorithms.
In this particular case, we described an instance of *supervised learning* 
applied to a problem in automated speech recognition. 

Machine Learning is a versatile set of tools that lets you work with data in many different situations where simple rule-based systems would fail or might be very difficult to build. Due to its versatility, machine learning can be quite confusing to newcomers.
For example, machine learning techniques are already widely used
in applications as diverse as search engines, self driving cars, 
machine translation, medical diagnosis, spam filtering, 
game playing (*chess*, *go*), face recognition, 
data matching, calculating insurance premiums, and adding filters to photos. 

Despite the superficial differences between these problems many of them share a common structure
and are addressable with deep learning tools. 
They're mostly similar because they are problems where we wouldn't be able to program their behavior directly in code, 
but we can *program them with data*.
Often times the most direct language for communicating these kinds of programs is *math*. 
In this book, we'll introduce a minimal amount of mathematical notation,
but unlike other books on machine learning and neural networks,
we'll always keep the conversation grounded in real examples and real code.



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

## Supervised learning 

Supervised learning addresses the task of predicting *targets* given input data.
The targets, also commonly called *labels* are generally denoted *y*.
The input data points, also commonly called *examples* or *instances*, are typically denoted $\boldsymbol{x}$.
The goal is to produce a model $f_\theta$ that maps an input $\boldsymbol{x}$ to a prediction $f_{\theta}(\boldsymbol{x})$

To ground this description in a concrete example,
if we were working in healthcare,
then we might want to predict whether or not a patient would have a heart attack. 
This observation, *heart attack* or *no heart attack*,
would be our label $y$.
The input data $\boldsymbol{x}$ might be vital signs such as heart rate, diastolic and systolic blood pressure, etc. 

The supervision comes into play because for choosing the parameters $\theta$, we (the supervisors) provide the model with a collection of *labeled examples* ($\boldsymbol{x}_i, y_i$), where each example $\boldsymbol{x}_i$ is matched up against it's correct label.

In probabilistic terms, we typically are interested estimating 
the conditional probability $P(y|x)$. 
While it's just one among several approaches to machine learning, 
supervised learning accounts for the majority of machine learning in practice. 
Partly, that's because many important tasks 
can be described crisply as estimating the probability of some unknown given some available evidence: 

* **Predict cancer vs not cancer, given a CT image.** 
* **Predict the correct translation in French, given a sentence in English.**
* **Predict the price of a stock next month based on this month's financial reporting data.**

Even with the simple description "predict targets from inputs" 
supervised learning can take a great many forms and require a great many modeling decisions,
depending on the type, size, and the number of inputs and outputs. 
For example, we use different models to process sequences (like strings of text or time series data)
and for processing fixed-length vector representations.
We'll visit many of these problems in depth throughout the first 9 parts of this book. 

Put plainly, the learning process looks something like this.
Grab a big pile of example inputs, selecting them randomly.
Acquire the ground truth labels for each.
Together, these inputs and corresponding labels (the desired outputs)
comprise the training set. 
We feed the training dataset into a supervised learning algorithm.
So here the *supervised learning algorithm* is a function that takes as input a dataset,
and outputs another function, *the learned model*. 
Then, given a learned model, 
we can take a new previously unseen input, and predict the corresponding label.

![](../img/supervised-learning.png)



### Regression

Perhaps the simplest supervised learning task to wrap your head around is Regression.
Consider, for example a set of data harvested 
from a database of home sales.
We might construct a table, where each row corresponds to a different house, 
and each column corresponds to some relevant attribute,
such as the square footage of a house, the number of bedrooms, the number of bathrooms,
and the number of minutes (walking) to the center of town. 
Formally, we call one row in this dataset a *feature vector*,
and the object (e.g. a house) it's associated with an *example*.

If you live in New York or San Francisco, and you are not the CEO of Amazon, Google, Microsoft, or Facebook, 
the (sq. footage, no. of bedrooms, no. of bathrooms, walking distance) feature vector for your home 
might look something like: $[100, 0, .5, 60]$. 
However, if you live in Pittsburgh, 
it might look more like $[3000, 4, 3, 10]$.
Feature vectors like this are essential for all the classic machine learning problems.
We'll typically denote the feature vector for any one example $\mathbf{x_i}$
and the set of feature vectors for all our examples $X$.

What makes a problem *regression* is actually the outputs.
Say that you're in the market for a new home, 
you might want to estimate the fair market value of a house,
given some features like these. 
The target value, the price of sale, is a *real number*.
We denote any individual target $y_i$ (corresponding to example $\mathbf{x_i}$) 
and the set of all targets $\mathbf{y}$ (corresponding to all examples X). 
When our targets take on arbitrary real values in some range, 
we call this a regression problem. 
The goal of our model is to produce predictions (guesses of the price, in our example)
that closely approximate the actual target values.  
We denote these predictions $\hat{y}_i$ 
and if the notation seems unfamiliar, then just ignore it for now. 
We'll unpack it more thoroughly in the subsequent chapters.


Lots of practical problems are well-described regression problems. 
Predicting the rating that a user will assign to a movie is a regression problem,
and if you designed a great algorithm to accomplish this feat in 2009,
you might have won the [$1 million Netflix prize](https://en.wikipedia.org/wiki/Netflix_Prize).
Predicting the length of stay for patients in the hospital is also a regression problem.
A good rule of thumb is that any *How much?* or *How many?* problem should suggest regression.
* "How many hours will this surgery take?"... *regression*
* "How many dogs are in this photo?" ... *regression*.
However, if you can easily pose your problem as "Is this a ___?", then it's likely, classification, a different fundamental problem type that we'll cover next.

Even if you've never worked with machine learning before, 
you've probably worked through a regression problem informally. 
Imagine, for example, that you had your drains repaired 
and that your contractor, spent $x_1=3$ hours removing gunk from your sewage pipes.
Then she sent you a bill of $y_1 = \$350$. 
Now imagine that your friend hired the same contractor for $x_2 = 2$ hours 
and that she received a bill of $y_2 = \$250$. 
If someone then asked you how much to expect on their upcoming gunk-removal invoice
you might make some reasonable assumptions, 
such as more hours worked costs more dollars.
You might also assume that there's some base charge and that the contractor then charges per hour.
If these assumptions held, then given these two data points, 
you could already identify the contractor's pricing structure: 
\$100 per hour plus \$50 to show up at your house. 
If you followed that much then you already understand the high-level idea behind linear regression.

In this case, we could produce the parameters that exactly matched the contractor's prices. 
Sometimes that's not possible, e.g., if some of the variance owes to some factors besides your two features.
In these cases, we'll try to learn models that minimize the distance between our predictions and the observed values.
In most of our chapters, we'll focus on one of two very common losses, the [L1 loss](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L1Loss) where $l(y,y') = \sum_i |y_i-y_i'|$ and the [L2 loss](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L2Loss) where $l(y,y') = \sum_i (y_i - y_i')^2$.
As we will see later, the $L_2$ loss corresponds to the assumption that our data was corrupted by Gaussian noise, whereas the $L_1$ loss corresponds to an assumption of noise from a Laplace distribution. 

### Classification

While regression models are great for addressing *how many?* questions, 
lots of problems don't bend comfortably to this template. For example, 
a bank wants to add check scanning to their mobile app. 
This would involve the customer snapping a photo of a check with their smartphone's camera 
and the machine learning model would need to be able to automatically understand text seen in the image. 
It would also need to understand hand-written text to be even more robust. 
This kind of system is referred to as optical character recognition (OCR), 
and the kind of problem it solves is called a classification. 
It's treated with a distinct set of algorithms than those that are used for regression. 

In classification, we want to look at a feature vector, like the pixel values in an image, 
and then predict which category (formally called *classes*), 
among some set of options, an example belongs.
For hand-written digits, we might have 10 classes, 
corresponding to the digits 0 through 9.
The simplest form of classification is when there are only two classes, 
a problem which we call binary classification.
For example, our dataset $X$ could consist of images of animals 
and our *labels* $Y$ might be the classes $\mathrm{\{cat, dog\}}$.
While in regression, we sought a regressor to output a real value $\hat{y}$,
in classification, we seek a *classifier*, whose output $\hat{y}$ is the predicted class assignment.

For reasons that we'll get into as the book gets more technical, it's pretty hard to optimize a model that can only output a hard categorical assignment, e.g. either *cat* or *dog*. 
It's a lot easier instead to express the model in the language of probabilities. 
Given an example $x$, the model assigns a probability $\hat{y}_k$ to each label $k$. 
Because these are probabilities, they need to be positive numbers and add up to $1$. 
This means that we only need $K-1$ numbers to give the probabilities of $K$ categories.
This is easy to see for binary classification. 
If there's a 0.6 (60%) probability that an unfair coin comes up heads, 
then there's a 0.4 (40%) probability that it comes up tails. 
Returning to our animal classification example, a classifier might see an image 
and output the probability that the image is a cat $\Pr(y=\mathrm{cat}\mid x) = 0.9$.
We can interpret this number by saying that the classifier is 90% sure that the image depicts a cat. 
The magnitude of the probability for the predicted class is one notion of confidence. 
It's not the only notion of confidence and we'll discuss different notions of uncertainty in more advanced chapters.

When we have more than two possible classes, we call the problem *multiclass classification*.
Common examples include hand-written character recognition `[0, 1, 2, 3 ... 9, a, b, c, ...]`. 
While we attacked regression problems by trying to minimize the L1 or L2 loss functions,
the common loss function for classification problems is called cross-entropy.
In `MXNet Gluon`, the corresponding loss function can be found [here](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss). 

Note that the most likely class is not necessarily the one that you're going to use for your decision. Assume that you find this beautiful mushroom in your backyard:

|![](../img/death_cap.jpg)|
|:-------:|
|Death cap - do not eat!|

Now, assume that you built a classifier and trained it 
to predict if a mushroom is poisonous based on a photograph.
Say our poison-detection classifier outputs $\Pr(y=\mathrm{death cap}\mid\mathrm{image}) = 0.2$. 
In other words, the classifier is 80% confident that our mushroom *is not* a death cap. 
Still, you'd have to be a fool to eat it. 
That's because the certain benefit of a delicious dinner isn't worth a 20% chance of dying from it. 
In other words, the effect of the *uncertain risk* by far outweighs the benefit. 
Let's look at this in math. Basically, we need to compute the expected risk that we incur, i.e. we need to multiply the probability of the outcome with the benefit (or harm) associated with it:

$$L(\mathrm{action}\mid x) = \mathbf{E}_{y \sim p(y\mid x)}[\mathrm{loss}(\mathrm{action},y)]$$

Hence, the loss $L$ incurred by eating the mushroom is $L(a=\mathrm{eat}\mid x) = 0.2 * \infty + 0.8 * 0 = \infty$, whereas the cost of discarding it is $L(a=\mathrm{discard}\mid x) = 0.2 * 0 + 0.8 * 1 = 0.8$. 

We got lucky: as any mycologist would tell us, the above actually *is* a death cap.
Classification can get much more complicated than just binary, multiclass, of even multi-label classification.
For instance, there are some variants of classification for addressing hierarchies. 
Hierarchies assume that there exist some relationships among the many classes.
So not all errors are equal - we prefer to misclassify to a related class than to a distant class.
Usually, this is referred to as *hierarchical classification*. 
One early example is due to [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus),
who organized the animals in a hierarchy.. 

![](../img/taxonomy.jpg)

In the case of animal classification, it might not be so bad to mistake a poodle for a schnauzer, 
but our model would pay a huge penalty if it confused a poodle for a dinosaur. 
What hierarchy is relevant might depend on how you plan to use the model.
For example, rattle snakes and garter snakes might be close on the phylogenetic tree, 
but mistaking a rattler for a garter could be deadly. 

### Tagging

Some classification problems don't fit neatly into the binary or multiclass classification setups. 
For example, we could train a normal binary classifier to distinguish cats from dogs.
Given the current state of computer vision, 
we can do this easily, with off-the-shelf tools.
Nonetheless, no matter how accurate our model gets, we might find ourselves in trouble when the classifier encounters an image like this: 

![](../img/catdog.jpg)

As you can see, there's a cat in the picture. 
There is also a dog, a tire, some grass, a door, concrete, 
rust, individual grass leaves, etc. 
Depending on what we want to do with our model ultimately, 
treating this as a binary classification problem 
might not make a lot of sense. 
Instead, we might want to give the model the option 
of saying the image depicts a cat *and* a dog, 
or *neither* a cat *nor* a dog. 

The problem of learning to predict classes 
that are *not mutually exclusive* 
is called multi-label classification. 
Auto-tagging problems are typically best described 
as multi-label classification problems. 
Think of the tags people might apply to posts on a tech blog, 
e.g., "machine learning", "technology", "gadgets", 
"programming languages", "linux", "cloud computing", "AWS". 
A typical article might have 5-10 tags applied 
because these concepts are correlated. 
Posts about "cloud computing" are likely to mention "AWS" 
and posts about "machine learning" could also deal with "programming languages". 

We also have to deal with this kind of problem when dealing with the biomedical literature,
where correctly tagging articles is important 
because it allows researchers to do exhaustive reviews of the literature. 
At the National Library of Medicine, a number of professional annotators 
go over each article that gets indexed in PubMed 
to associate each with the relevant terms from MeSH, 
a collection of roughly 28k tags. 
This is a time-consuming process and the annotators typically have a one year lag between archiving and tagging. Machine learning can be used here to provide provisional tags 
until each article can have a proper manual review. 
Indeed, for several years, the BioASQ organization has [hosted a competition](http://bioasq.org/) 
to do precisely this.


### Search and ranking

Sometimes we don't just want to assign each example to a bucket or to a real value. In the field of information retrieval, we want to impose a ranking on a set of items. Take web search for example, the goal is less to determine whether a particular page is relevant for a query, but rather, which one of the plethora of search results should be displayed for the user. We really care about the ordering of the relevant search results and our learning algorithm needs to produce ordered subsets of elements from a larger set. In other words, if we are asked to produce the first 5 letters from the alphabet, there is a difference between returning ``A B C D E`` and ``C A B E D``. Even if the result set is the same, the ordering within the set matters nonetheless.

One possible solution to this problem is to score every element in the set of possible sets along with a corresponding relevance score and then to retrieve the top-rated elements. [PageRank](https://en.wikipedia.org/wiki/PageRank) is an early example of such a relevance score. One of the peculiarities is that it didn't depend on the actual query. Instead, it simply helped to order the results that contained the query terms. Nowadays search engines use machine learning and behavioral models to obtain query-dependent relevance scores. There are entire conferences devoted to this subject. 

<!-- Add / clean up-->

### Recommender systems

Recommender systems are another problem setting that is related to search and ranking. The problems are  similar insofar as the goal is to display a set of relevant items to the user. The main difference is the emphasis on *personalization* to specific users in the context of recommender systems. For instance, for movie recommendations, the results page for a SciFi fan and the results page for a connoisseur of Woody Allen comedies might differ significantly. 

Such problems occur, e.g. for movie, product or music recommendation. In some cases, customers will provide explicit details about how much they liked the product (e.g. Amazon product reviews). In some other cases, they might simply provide feedback if they are dissatisfied with the result (skipping titles on a playlist). Generally, such systems strive to estimate some score $y_{ij}$, such as an estimated rating or probability of purchase, given a user $u_i$ and product $p_j$. 

Given such a model, then for any given user, we could retrieve the set of objects  with the largest scores $y_{ij}$ are then used as a recommendation. Production systems are considerably more advanced and take detailed user activity and item characteristics into account when computing such scores. The following image is an example of deep learning books recommended by Amazon based on personalization algorithms tuned to the author's preferences.

![](../img/deeplearning_amazon.png)


### Sequence Learning

So far we've looked at problems where we have some fixed number of inputs 
and produce a fixed number of outputs. 
Before we considered predicting home prices from a fixed set of features:
square footage, number of bedrooms, number of bathrooms, walking time to downtown.
We also discussed mapping from an image (of fixed dimension), 
to the predicted probabilities that it belongs to each of a fixed number of classes,
or taking a user ID and a product ID, and predicting a star rating.
In these cases, once we feed our fixed-length input into the model to generate an output, 
the model immediately forgets what it just saw. 

This might be fine if our inputs truly all have the same dimensions 
and if successive inputs truly have nothing to do with each other. 
But how would we deal with video snippets?
In this case, each snippet might consist of a different number of frames.
And our guess of what's going on in each frame
might be much stronger if we take into account 
the previous or succeeding frames. 
Same goes for language. 
One popular deep learning problem is machine translation:
the task of ingesting sentences in some source language 
and predicting their translation in another language.

These problems also occur in medicine.
We might want a model to monitor patients in the intensive care unit and to fire off alerts 
if their risk of death in the next 24 hours exceeds some threshold. 
We definitely wouldn't want this model to throw away everything it knows about the patient history each hour, 
and just make its predictions based on the most recent measurements. 

These problems are among the more exciting applications of machine learning
and they are instances of *sequence learning*. 
They require a model to either ingest sequences of inputs 
or to emit sequences of outputs (or both!).
These latter problems are sometimes referred to as ``seq2seq`` problems. 
Language translation is a ``seq2seq`` problem. 
Transcribing text from spoken speech is also a ``seq2seq`` problem.
While it is impossible to consider all types of sequence transformations, 
a number of special cases are worth mentioning:

#### Tagging and Parsing

This involves annotating a text sequence with attributes. In other words, the number of inputs and outputs is essentially the same. For instance, we might want to know where the verbs and subjects are. Alternatively, we might want to know which words are the named entities. In general, the goal is to decompose and annotate text based on structural and grammatical assumptions to get some annotation. This sounds more complex than it actually is. Below is a very simple example of annotating a sentence with tags indicating which words refer to named entities. 


|Tom | wants | to | have | dinner | in | Washington | with | Sally.|
|:--|
|Ent | - | - | - | - | - | Ent | - | Ent|


#### Automatic Speech Recognition

With speech recognition, the input sequence $x$ is the sound of a speaker, 
and the output $y$ is the textual transcript of what the speaker said. 
The challenge is that there are many more audio frames (sound is typically sampled at 8kHz or 16kHz) than text, i.e. there is no 1:1 correspondence between audio and text,
since thousands of samples correspond to a single spoken word. 
These are seq2seq problems where the output is much shorter than the input. 

|`----D----e----e-----p------- L----ea------r------ni-----ng---`|
|:--------------|
|![Deep Learning](../img/speech.jpg)|

#### Text to Speech

Text to Speech (TTS) is the inverse of speech recognition.
In other words, the input $x$ is text 
and the output $y$ is an audio file. 
In this case, the output is *much longer* than the input. 
While it is easy for *humans* to recognize a bad audio file, 
this isn't quite so trivial for computers. 

#### Machine Translation

Unlike the case of speech of recognition, where corresponding inputs and outputs occur in the same order (after alignment), 
in machine translation, order inversion can be vital. 
In other words, while we are still converting one sequence into another, 
neither the number of inputs and outputs 
nor the order of corresponding data points 
are assumed to be the same.
Consider the following illustrative example of the obnoxious tendency of Germans 
(*Alex writing here*) 
to place the verbs at the end of sentences. 

|German |Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?|
|:------|:---------|
|English|Did you already check out this excellent tutorial?|
|Wrong alignment |Did you yourself already this excellent tutorial looked-at?|

A number of related problems exist. 
For instance, determining the order in which a user reads a webpage 
is a two-dimensional layout analysis problem. 
Likewise, for dialogue problems, 
we need to take world-knowledge and prior state into account. 
This is an active area of research.


## Unsupervised learning

All the examples so far were related to *Supervised Learning*, 
i.e. situations where we feed the model 
a bunch of examples and a bunch of *corresponding target values*.
You could think of supervised learning as having an extremely specialized job and an extremely anal boss. 
The boss stands over your shoulder and tells you exactly what to do in every situation until you learn to map from situations to actions.
Working for such a boss sounds pretty lame. 
On the other hand, it's easy to please this boss. You just recognize the pattern as quickly as possible and imitate their actions. 

In a completely opposite way,
it could be frustrating to work for a boss 
who has no idea what they want you to do.
However, if you plan to be a data scientist, you had better get used to it. 
The boss might just hand you a giant dump of data and tell you to *do some data science with it!*
This sounds vague because it is. 
We call this class of problems *unsupervised learning*, 
and the type and number of questions we could ask 
is limited only by our creativity. 
We will address a number of unsupervised learning techniques in later chapters. To whet your appetite for now, we describe a few of the questions you might ask:

* Can we find a small number of prototypes that accurately summarize the data? Given a set of photos, can we group them into landscape photos, pictures of dogs, babies, cats, mountain peaks, etc.? Likewise, given a collection of users' browsing activity, can we group them into users with similar behavior? This problem is typically known as **clustering**.
* Can we find a small number of parameters that accurately capture the relevant properties of the data? The trajectories of a ball are quite well described by velocity, diameter, and mass of the ball. Tailors have developed a small number of parameters that describe human body shape fairly accurately for the purpose of fitting clothes. These problems are referred to as **subspace estimation** problems. If the dependence is linear, it is called **principal component analysis**.
* Is there a representation of (arbitrarily structured) objects in Euclidean space (i.e. the space of vectors in $\mathbb{R}^n$) such that symbolic properties can be well matched? This is called **representation learning** and it is used to describe entities and their relations, such as Rome - Italy + France = Paris. 
* Is there a description of the root causes of much of the data that we observe? For instance, if we have demographic data about house prices, pollution, crime, location, education, salaries, etc., can we discover how they are related simply based on empirical data? The field of **directed graphical models** and **causality** deals with this.
* An important and exciting recent development is **generative adversarial networks**. They are basically a procedural way of synthesizing data. The underlying statistical mechanisms are tests to check whether real and fake data are the same. We will devote a few notebooks to them. 


## Interacting with an environment

So far, we haven't discussed where data actually comes from, 
or what actually *happens* when a machine learning model generates an output. 
That's because supervised learning and unsupervised learning
do not address these issues in a very sophisticated way.
In either case, we grab a big pile of data up front,
then do our pattern recognition without ever interacting with the environment again.
Because all of the learning takes place after the algorithm is disconnected from the environment,
this is called *offline learning*.
For supervised learning, the process looks like this:

![](../img/data-collection.png)


This simplicity of offline learning has its charms.
The upside is we can worry about pattern recognition in isolation without these other problems to deal with,
but the downside is that the problem formulation is quite limiting.
If you are more ambitious, or if you grew up reading Asimov's Robot Series,
then you might imagine artificially intelligent bots capable not only of making predictions,
but of taking actions in the world. 
We want to think about intelligent *agents*, not just predictive *models*. 
That means we need to think about choosing *actions*, not just making *predictions*.
Moreover, unlike predictions, actions actually impact the environment. 
If we want to train an intelligent agent,
we must account for the way its actions might 
impact the future observations of the agent. 


Considering the interaction with an environment that opens a whole set of new modeling questions. Does the environment:

* remember what we did previously?
* want to help us, e.g. a user reading text into a speech recognizer?
* want to beat us, i.e. an adversarial setting like spam filtering (against spammers) or playing a game (vs an opponent)?
* not  care (as in most cases)?
* have shifting dynamics (steady vs shifting over time)?

This last question raises the problem of *covariate shift*,
(when training and test data are different). 
It's a problem that most of us have experienced when taking exams written by a lecturer,
while the homeworks were composed by his TAs. 
We'll briefly describe reinforcement learning, and adversarial learning, 
two settings that explicitly consider interaction with an environment. 


### Reinforcement learning

If you're interested in using machine learning to develop an agent that interacts with an environment and takes actions, then you're probably going to wind up focusing on *reinforcement learning* (RL). 
This might include applications to robotics, to dialogue systems, 
and even to developing AI for video games. 
*Deep reinforcement learning* (DRL), which applies deep neural networks
to RL problems, has surged in popularity. 
The breakthrough [deep Q-network that beat humans at Atari games using only the visual input](https://www.wired.com/2015/02/google-ai-plays-atari-like-pros/) ,
and the [AlphaGo program that dethroned the world champion at the board game Go](https://www.wired.com/2017/05/googles-alphago-trounces-humans-also-gives-boost/) are two prominent examples.

Reinforcement learning gives a very general statement of a problem,
in which an agent interacts with an environment over a series of *time steps*.
At each time step $t$, the agent receives some observation $o_t$ from the environment,
and must choose an action $a_t$ which is then transmitted back to the environment. 
Finally, the agent receives a reward $r_t$ from the environment.
The agent then receives a subseqeunt observation, and chooses a subsequent action, and so on.
The behavior of an RL agent is governed by a *policy*.
In short, a *policy* is just a function that maps from observations (of the environment) to actions.
The goal of reinforcement learning is to produce a good policy.

![](../img/rl-environment.png)

It's hard to overstate the generality of the RL framework.
For example, we can cast any supervised learning problem as an RL problem. 
Say we had a classification problem. 
We could create an RL agent with one *action* corresponding to each class. 
We could then create an environment which gave a reward 
that was exactly equal to the loss function from the original supervised problem.

That being said, RL can also address many problems that supervised learning cannot. 
For example, in supervised learning we always expect
that the training input comes associated with the correct label.
But in RL, we don't assume that for each observation, 
the environment tells us the optimal action.
In general, we just get some reward.
Moreover, the environment may not even tell us which actions led to the reward. 

Consider for example the game of chess. 
The only real reward signal comes at the end of the game when we either win, which we might assign a reward of 1,
or when we lose, which we could assign a reward of -1.
So reinforcement learners must deal with the *credit assignment problem*.
The same goes for an employee who gets a promotion on October 11. 
That promotion likely reflects a large number of well-chosen actions over the previous year.
Getting more promotions in the future requires figuring out what actions along the way led to the promotion.

Reinforcement learners may also have to deal with the problem of partial observability. 
That is, the current observation might not tell you everything about your current state. 
Say a cleaning robot found itself trapped in one of many identical closets in a house.
Inferring the precise location (and thus state) of the robot
might require considering its previous observerations before entering the closet. 

Finally, at any given point, reinforcement learners might know of one good policy,
but there might be many other better policies that the agent has never tried. 
The reinforcement learner must constantly choose 
whether to *exploit* the best currently-known strategy as a policy,
or to *explore* the space of strategies, 
potentially giving up some short-run reward in exchange for knowledge.


### MDPs, bandits, and friends

The general reinforcement learning problem
is a very general setting. 
Actions affect subsequent observations. 
Rewards are only observed corresponding to the chosen actions.
The environment may be either fully or partially observed.
Accounting for all this complexity at once may ask too much of researchers.
Moreover not every practical problem exhibits all this complexity.
As a result, researchers have studied a number of *special cases* of reinforcement learning problems. 

When the environment is fully observed, we call the RL problem a *Markov Decision Process* (MDP).
When the state does not depend on the previous actions, 
we call the problem a *contextual bandit problem*. 
When there is no state, just a set of available actions with initially unknown rewards,
this problem is the classic *multi-armed bandit problem*. 

## When *not* to use machine learning

Let's take a closer look at the idea of programming data
by considering an interaction that [Joel Grus](http://joelgrus.com) reported experiencing in a [job interview](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/). The interviewer asked him to code up Fizz Buzz. This is a children's game where the players count from 1 to 100 and will say *'fizz'* whenever the number is divisible by 3, *'buzz'* whenever it is divisible by 5, and *'fizzbuzz'* whenever it satisfies both criteria. Otherwise, they will just state the number. It looks like this:

```
1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 fizz 13 14 fizzbuzz 16 ...
```

The conventional way to solve such a task is quite simple.


```python
res = []
for i in range(1, 101):
    if i % 15 == 0:
        res.append('fizzbuzz')
    elif i % 3 == 0:
        res.append('fizz')
    elif i % 5 == 0:
        res.append('buzz')
    else:
        res.append(str(i))
print(' '.join(res))
```

    1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 fizz 13 14 fizzbuzz 16 17 fizz 19 buzz fizz 22 23 fizz buzz 26 fizz 28 29 fizzbuzz 31 32 fizz 34 buzz fizz 37 38 fizz buzz 41 fizz 43 44 fizzbuzz 46 47 fizz 49 buzz fizz 52 53 fizz buzz 56 fizz 58 59 fizzbuzz 61 62 fizz 64 buzz fizz 67 68 fizz buzz 71 fizz 73 74 fizzbuzz 76 77 fizz 79 buzz fizz 82 83 fizz buzz 86 fizz 88 89 fizzbuzz 91 92 fizz 94 buzz fizz 97 98 fizz buzz


This isn't very exciting if you're a good programmer. Joel proceeded to 'implement' this problem in Machine Learning instead. For that to succeed, he needed a number of pieces:

* Data X ``[1, 2, 3, 4, ...]`` and labels Y ``['fizz', 'buzz', 'fizzbuzz', identity]`` 
* Training data, i.e. examples of what the system is supposed to do. Such as ``[(2, 2), (6, fizz), (15, fizzbuzz), (23, 23), (40, buzz)]``
* Features that map the data into something that the computer can handle more easily, e.g. ``x -> [(x % 3), (x % 5), (x % 15)]``. This is optional but helps a lot if you have it. 

Armed with this, Joel wrote a classifier in TensorFlow ([code](https://github.com/joelgrus/fizz-buzz-tensorflow)). The interviewer was nonplussed ... and the classifier didn't have perfect accuracy.

Quite obviously, this is silly. Why would you go through the trouble of replacing a few lines of Python with something much more complicated and error prone? However, there are many cases where a simple Python script simply does not exist, yet a 3-year-old child will solve the problem perfectly. 
Fortunately, this is precisely where machine learning comes to the rescue. 

## Conclusion

Machine Learning is vast. We cannot possibly cover it all. On the other hand, neural networks are simple and only require elementary mathematics. So let's get started. 

## Next
[Manipulate data the MXNet way with NDArray](../chapter01_crashcourse/ndarray.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
