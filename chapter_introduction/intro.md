# Introduction
:label:`chapter_introduction`

Until recently, nearly all of the computer programs
that we interacted with every day were coded
by software developers from first principles.
Say that we wanted to write an application to manage an e-commerce platform.
After huddling around a whiteboard for a few hours to ponder the problem,
we would come up with the broad strokes of a working solution
that would probably look something like this:
(i) users would interact with the application
through an interface running in a web browser or mobile application
(ii) our application would rely on a commerical database engine
to keep track of each user's state and maintain records
of all historical transactions
(ii) at the heart of our application, running in parallel across many servers, the *business logic* (you might say, the *brains*)
would map out in methodical details the appropriate action to take
in every conceivable circumstance.

To build the *brains* of our application,
we'd have to step through every possible corner case
that we anticipate encountering, devising appropriate rules.
Each time a customer clicks to add an item to their shopping cart,
we add an entry to the shopping cart database table,
associating that user's ID with the requested product’s ID.
While few developers ever get it completely right the first time
(it might take some test runs to work out the kinks),
for the most part, we could write such a program from first principles
and confidently launch it *before ever seeing a real customer*.
Our ability to design automated systems from first principles
that drive functioning products and systems,
often in novel situations, is a remarkable cognitive feat.
And when you're able to devise solutions that work $100\%$ of the time,
*you should not be using machine learning*.

Fortunately—for the growing community of ML scientists—many
problems in automation don't bend so easily to human ingenuity.
Imagine huddling around the whiteboard with the smartest minds you know,
but this time you are tackling any of the following problems:

* Write a program that predicts tomorrow's weather given geographic
information, satellite images, and a trailing window of past weather.
* Write a program that takes in a question, expressed in free-form text, and
 answers it correctly.
* Write a program that given an image can identify all the people it contains,
 drawing outlines around each.
* Write a program that presents users with products that they are likely to
  enjoy but unlikely, in the natural course of browsing, to encounter.

In each of these cases, even elite programmers
are incapable of coding up solutions from scratch.
The reasons for this can vary.
Sometimes the program that we are looking for
follows a pattern that changes over time,
and we need our programs to adapt.
In other cases, the relationship
(say between pixels, and abstract categories)
may be too complicated, requiring thousands or millions of computations
that are beyond our conscious understanding
(even if our eyes manage the task effortlessly).
Machine learning (ML) is the study of powerful techniques
that can *learn behavior* from *experience*.
As ML algorithm accumulates more experience,
typically in the form of observational data
or interactions with an environment, their performance improves.
Contrast this with our deterministic e-commerce platform,
which performs according to the same business logic,
no matter how much experience accrues,
until the developers themselves *learn* and decide
that it's time to update the software.
In this book, we will teach you the fundamentals of machine learning,
and focus in particular on deep learning,
a powerful set of techniques driving innovations
in areas as diverse as computer vision, natural language processing,
healthcare, and genomics.

## A Motivating Example

Before we could begin writing, the authors of this book,
like much of the work force, had to become caffeinated.
We hopped in the car and started driving.
Using an iPhone, Alex called out 'Hey Siri',
awakening the phone's voice recognition system.
Then Mu commanded 'directions to Blue Bottle coffee shop'.
The phone quickly displayed the transcription of his command.
It also recognized that we were asking for directions
and launched the Maps application to fulfill our request.
Once launched, the Maps app identified a number of routes.
Next to each route, the phone displayed a predicted transit time.
While we fabricated this story for pedagogical convenience,
it demonstrates that in the span of just a few seconds,
our everyday interactions with a smartphone
can engage several machine learning models.

Imagine just writing a program to respond to a *wake word*
like 'Alexa', 'Okay, Google' or 'Siri'.
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
That's why we use ML.

![Identify an awake word.](../img/wake-word.svg)


Here's the trick.
Often, even when we don't know how to tell a computer
explicitly how to map from inputs to outputs,
we are nonetheless capable of performing the cognitive feat ourselves.
In other words, even if you don't know *how to program a computer*
to recognize the word 'Alexa',
you yourself *are able* to recognize the word 'Alexa'.
Armed with this ability,
we can collect a huge *dataset* containing examples of audio
and label those that *do* and that *do not* contain the wake word.
In the ML approach, we do not design a system *explicitly*
to recognize wake words.
Instead, we define a flexible program
whose behavior is determined by a number of *parameters*.
Then we use the dataset to determine
the best possible set of parameters,
those that improve the performance of our program
with respect to some measure of performance on the task of interest.

You can think of the parameters as knobs that we can turn,
manipulating the behavior of the program.
Fixing the parameters, we call the program a *model*.
The set of all distinct programs (input-output mappings)
that we can produce just by manipulating the parameters
is called a *family* of models.
And the *meta-program* that uses our dataset
to choose the parameters is called a *learning algorithm*.

Before we can go ahead and engage the learning algorithm,
we have to define the problem precisely,
pinning down the exact nature of the inputs and outputs,
and choosing an appropriate model family.
In this case, our model receives a snippet of audio as *input*,
and it generates a selection among ``{yes, no}`` as *output*—which,
if all goes according to plan,
will closely approximate whether (or not)
the snippet contains the wake word.

If we choose the right family of models,
then there should exist one setting of the knobs
such that the model fires ``yes`` every time it hears the word 'Alexa'.
Because the exact choice of the wake word is arbitrary,
we'll probably need a model family capable, via another setting of the knobs,
of firing ``yes`` on the word 'Apricot'.
We expect that the same model should apply to 'Alexa' recognition and 'Apricot' recognition because these are similar tasks.
However, we might need a different family of models entirely
if we want to deal with fundamentally different inputs or outputs,
say if we wanted to map from images to captions,
or from English sentences to Chinese sentences.

As you might guess, if we just set all of the knobs randomly,
it's not likely that our model will recognize 'Alexa', 'Apricot',
or any other English word.
In deep learning, the *learning* is the process
by which we discover the right setting of the knobs
coercing the desired behaviour from our model.

The training process usually looks like this:

1. Start off with a randomly initialized model that can't do anything useful.
1. Grab some of your labeled data (e.g. audio snippets and corresponding ``{yes,no}`` labels)
1. Tweak the knobs so the model sucks less with respect to those examples
1. Repeat until the model is awesome.

![A typical training process. ](../img/ml-loop.svg)

To summarize, rather than code up a wake word recognizer,
we code up a program that can *learn* to recognize wake words,
*if we present it with a large labeled dataset*.
You can think of this act
of determining a program's behavior by presenting it with a dataset
as *programming with data*.
We can "program" a cat detector by providing our machine learning system
with many examples of cats and dogs, such as the images below:

| ![cat1](../img/cat1.png) | ![cat2](../img/cat2.jpg) | ![dog1](../img/dog1.jpg) |![dog2](../img/dog2.jpg) |
|:---------------:|:---------------:|:---------------:|:---------------:|
|cat|cat|dog|dog|

This way the detector will eventually learn to emit
a very large positive number if it's a cat,
a very large negative number if it's a dog,
and something closer to zero if it isn't sure,
and this barely scratches the surface of what ML can do.

Deep learning is just one among many
popular frameworks for solving machine learning problems.
While thus far, we've only talked about machine learning broadly
and not deep learning, there's a couple points worth sneaking in here:
First, the problems that we've discussed thus far:
learning from raw audio signal,
directly from the pixels in images,
and mapping between sentences of arbitrary lengths and across languages
are problems where deep learning excels and traditional ML tools faltered.
Deep models are *deep* in precisely the sense that they learn
many *layers* of computation.
It turns out that these many-layered (or hierarchical) models
are capable of addressing low-level perceptual data
in a way that previous tools could not.
In bygone days, the crucial part of applying ML to these problems
consisted of coming up with manually engineered ways of transforming
the data into some form amenable to *shallow* models.
One key advantage of deep learning is that it replaces not only the *shallow* models
at the end of traditional learning pipelines,
but also the labor-intensive feature engineering.
Secondly, by replacing much of the *domain-specific preprocessing*,
deep learning has eliminated many of the boundaries
that previously separated computer vision, speech recognition,
natural language processing, medical informatics, and other application areas,
offering a unified set of tools for tackling diverse problems.


## The Key Components: Data, Models, and Algorithms

In our *wake-word* example, we described a dataset
consisting of audio snippets and binary labels
gave a hand-wavy sense of how we might *train*
a model to approximate a mapping from snippets to classifications.
This sort of problem, where we try to predict a designated unknown *label*
given known *inputs* (also called *features* or *covariates*),
and examples of both is called *supervised learning*,
and it's just one among many *kinds* of machine learning problems.
In the next section, we'll take a deep dive into the different ML problems.
First, we'd like to shed more light on some core components
that will follow us around, no matter what kind of ML problem we take on:

1. The **data** that we can learn from
2. A **model** of how to transform the data
3. A **loss** function that quantifies the *badness* of our model
4. An **algorithm** to adjust the model's parameters to minimize the loss


### Data

It might go without saying that you cannot do data science without data.
We could lose hundreds of pages pondering the precise nature of data
but for now we'll err on the practical side and focus on the key properties
to be concerned with.
Generally we are concerned with a collection of *examples*
(also called *data points*, *samples*, or *instances*).
In order to work with data usefully, we typically
need to come up with a suitable numerical representation.
Each *example* typically consists of a collection
of numerical attributes called *features* or *covariates*.

If we were working with image data,
each individual photograph might constitute an *example*,
each represented by an ordered list of numerical values
corresponding to the brightness of each pixel.
A $200\times200$ color photograph would consist of $200\times200\times3=120000$
numerical values, corresponding to the brightness
of the red, green, and blue channels corresponding to each spatial location.
In a more traditional task, we might try to predict
whether or not a patient will survive,
given a standard set of features such as age, vital signs, diagnoses, etc.

When every example is characterized by the same number of numerical values,
we say that the data consists of *fixed-length* vectors
and we describe the (constant) length of the vectors
as the *dimensionality* of the data.
As you might imagine, fixed length can be a convenient property.
If we wanted to train a model to recognize cancer in microscopy images,
fixed-length inputs means we have one less thing to worry about.

However, not all data can easily be represented as fixed length vectors.
While we might expect microscrope images to come from standard equipment,
we can't expect images mined from the internet to all show up in the same size.
While we might imagine cropping images to a standard size,
text data resists fixed-length representations even more stubbornly.
Consider the product reviews left on e-commerce sites like Amazon or TripAdvisor. Some are short: "it stinks!". Others ramble for pages.
One major advantage of deep learning over traditional methods
is the comparative grace with which modern models
can handle *varying-length* data.

Generally, the more data we have, the easier our job becomes.
When we have more data, we can train more powerful models,
and rely less heavily on pre-conceived assumptions.
The regime change from (comparatively small) to big data
is a major contributor to the success of modern deep learning.
To drive the point home, many of the most exciting models in deep learning either don't work without large data sets.
Some others work in the low-data regime,
but no better than traditional approaches.

Finally it's not enough to have lots of data and to process it cleverly.
We need the *right* data.
If the data is full of mistakes, or if the chosen features are not predictive of the target quantity of interest, learning is going to fail.
The situation is well captured by the cliché: *garbage in, garbage out*.
Moreover, poor predictive performance isn't the only potential consequence.
In sensitive applications of machine learning,
like predictive policing, resumé screening, and risk models used for lending,
we must be especially alert to the consequences of garbage data.
One common failure mode occurs in datasets where some groups of people
are unrepresented in the training data.
Imagine applying a skin cancer recognition system in the wild
that had never seen black skin before.
Failure can also occur when the data doesn't merely under-represent some groups,
but reflects societal prejudices.
For example if past hiring decisions are used to train a predictive model
that will be used to screen resumes, then machine learning models could inadvertently capture and automate historical injustices.
Note that this can all happen without the data scientist being complicit,
or even aware.


### Models


Most machine learning involves *transforming* the data in some sense.
We might want to build a system that ingests photos and predicts *smiley-ness*.
Alternatively, we might want to ingest a set of sensor readings
and predict how *normal* vs *anomalous* the readings are.
By *model*, we denote the computational machinery for ingesting data
of one type, and spitting out predictions of a possibly different type.
In particular, we are interested in statistical models
that can be estimated from data.
While simple models are perfectly capable of addressing
appropriately simple problems the problems
that we focus on in this book stretch the limits of classical methods.
Deep learning is differentiated from classical approaches
principally by the set of powerful models that it focuses on.
These models consist of many successive transformations of the data
that are chained together top to bottom, thus the name *deep learning*.
On our way to discussing deep neural networks, we'll discuss some more traditional methods.


###  Objective functions

Earlier, we introduced machine learning as "learning behavior from experience".
By *learning* here, we mean *improving* at some task over time.
But who is to say what constitutes an improvement?
You might imagine that we could propose to update our model,
and some people might disagree on whether the proposed update
constitued an improvement or a decline.

In order to develop a formal mathematical system of learning machines,
we need to have formal measures of how good (or bad) our models are.
In machine learning, and optimization more generally,
we call these objective functions.
By convention, we usually define objective funcitons
so that *lower* is *better*.
This is merely a convention. You can take any function $f$
for which higher is better, and turn it into a new function $f'$
that is qualitatively identical but for which lower is better
by setting $f' = -f$.
Because lower is better, these functions are sometimes called
*loss functions* or *cost functions*.

When trying to predict numerical values,
the most common objective function is squared error $(y-\hat{y})^2$.
For classification, the most common objective is to minimize error rate,
i.e., the fraction of instances on which
our predictions disagree with the ground truth.
Some objectives (like squared error) are easy to optimize.
Others (like error rate) are difficult to optimize directly,
owing to non-differentiability or other complications.
In these cases, it's common to optimize a surrogate objective.

Typically, the loss function is defined
with respect to the model's parameters
and depends upon the dataset.
The best values of our model's parameters are learned
by minimizing the loss incurred on a *training set*
consisting of some number of *examples* collected for training.
However, doing well on the training data
doesn't guarantee that we will do well on (unseen) test data.
So we'll typically want to split the available data into two partitions:
the training data (for fitting model parameters)
and the test data (which is held out for evaluation),
reporting the following two quantities:

 * **Training Error:**
 The error on that data on which the model was trained.
 You could think of this as being like
 a student's scores on practice exams
 used to prepare for some real exam.
 Even if the results are encouraging,
 that does not guarantee success on the final exam.
 * **Test Error:** This is the error incurred on an unseen test set.
 This can deviate significantly from the training error.
 When a model fails to generalize to unseen data,
 we say that it is *overfitting*.
 In real-life terms, this is like flunking the real exam
 despite doing well on practice exams.


### Optimization algorithms

Once we've got some data source and representation,
a model, and a well-defined objective function,
we need an algorithm capable of searching
for the best possible parameters for minimizing the loss function.
The most popular optimization algorithms for neural networks
follow an approach called gradient descent.
In short, at each step, they check to see, for each parameter,
which way the training set loss would move
if you perturbed that parameter just a small amount.
They then update the parameter in the direction that reduces the loss.


## Kinds of Machine Learning

In the following sections, we will discuss a few types of machine learning in some more detail. We begin with a list of *objectives*, i.e. a list of things that machine learning can do. Note that the objectives are complemented with a set of techniques of *how* to accomplish them, i.e. training, types of data, etc. The list below is really only sufficient to whet the readers' appetite and to give us a common language when we talk about problems. We will introduce a larger number of such problems as we go along.

### Supervised learning

Supervised learning addresses the task of predicting *targets* given input data.
The targets, also commonly called *labels*, are generally denoted *y*.
The input data points, also commonly called *examples* or *instances*, are typically denoted $\boldsymbol{x}$.
The goal is to produce a model $f_\theta$ that maps an input $\boldsymbol{x}$ to a prediction $f_{\theta}(\boldsymbol{x})$

To ground this description in a concrete example,
if we were working in healthcare,
then we might want to predict whether or not a patient would have a heart attack.
This observation, *heart attack* or *no heart attack*,
would be our label $y$.
The input data $\boldsymbol{x}$ might be vital signs such as heart rate, diastolic and systolic blood pressure, etc.

The supervision comes into play because for choosing the parameters $\theta$, we (the supervisors) provide the model with a collection of *labeled examples* ($\boldsymbol{x}_i, y_i$), where each example $\boldsymbol{x}_i$ is matched up against its correct label.

In probabilistic terms, we typically are interested in estimating
the conditional probability $P(y|x)$.
While it's just one among several approaches to machine learning,
supervised learning accounts for the majority of machine learning in practice.
Partly, that's because many important tasks
can be described crisply as estimating the probability of some unknown given some available evidence:

* Predict cancer vs not cancer, given a CT image.
* Predict the correct translation in French, given a sentence in English.
* Predict the price of a stock next month based on this month's financial reporting data.

Even with the simple description 'predict targets from inputs'
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

![Supervised learning.](../img/supervised-learning.svg)



#### Regression

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

What makes a problem a *regression* is actually the outputs.
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

* 'How many hours will this surgery take?' - *regression*
* 'How many dogs are in this photo?' - *regression*.

However, if you can easily pose your problem as 'Is this a _ ?',
then it's likely, classification, a different fundamental problem type that we'll cover next.
Even if you've never worked with machine learning before,
you've probably worked through a regression problem informally.
Imagine, for example, that you had your drains repaired
and that your contractor spent $x_1=3$ hours removing gunk from your sewage pipes.
Then she sent you a bill of $y_1 = \$350$.
Now imagine that your friend hired the same contractor for $x_2 = 2$ hours
and that she received a bill of $y_2 = \$250$.
If someone then asked you how much to expect on their upcoming gunk-removal invoice
you might make some reasonable assumptions,
such as more hours worked costs more dollars.
You might also assume that there's some base charge and that the contractor then charges per hour.
If these assumptions held true, then given these two data points,
you could already identify the contractor's pricing structure:
\$100 per hour plus \$50 to show up at your house.
If you followed that much then you already understand the high-level idea behind linear regression (and you just implicitly designed a linear model with bias).

In this case, we could produce the parameters that exactly matched the contractor's prices.
Sometimes that's not possible, e.g., if some of the variance owes to some factors besides your two features.
In these cases, we'll try to learn models that minimize the distance between our predictions and the observed values.
In most of our chapters, we'll focus on one of two very common losses,
the
[L1 loss](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L1Loss)
where

$$l(y,y') = \sum_i |y_i-y_i'|$$

and the least mean squares loss, aka
[L2 loss](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L2Loss)
where

$$l(y,y') = \sum_i (y_i - y_i')^2.$$

As we will see later, the $L_2$ loss corresponds to the assumption that our data was corrupted by Gaussian noise, whereas the $L_1$ loss corresponds to an assumption of noise from a Laplace distribution.

#### Classification

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
While in regression, we sought a *regressor* to output a real value $\hat{y}$,
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
and output the probability that the image is a cat $\Pr(y=\mathrm{cat}| x) = 0.9$.
We can interpret this number by saying that the classifier is 90% sure that the image depicts a cat.
The magnitude of the probability for the predicted class is one notion of confidence.
It's not the only notion of confidence and we'll discuss different notions of uncertainty in more advanced chapters.

When we have more than two possible classes, we call the problem *multiclass classification*.
Common examples include hand-written character recognition `[0, 1, 2, 3 ... 9, a, b, c, ...]`.
While we attacked regression problems by trying to minimize the L1 or L2 loss functions,
the common loss function for classification problems is called cross-entropy.
In MXNet Gluon, the corresponding loss function can be found [here](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss).

Note that the most likely class is not necessarily the one that you're going to use for your decision. Assume that you find this beautiful mushroom in your backyard:

![Death cap - do not eat!](../img/death_cap.jpg)
:width:`400px`

Now, assume that you built a classifier and trained it
to predict if a mushroom is poisonous based on a photograph.
Say our poison-detection classifier outputs $\Pr(y=\mathrm{death cap}|\mathrm{image}) = 0.2$.
In other words, the classifier is 80% confident that our mushroom *is not* a death cap.
Still, you'd have to be a fool to eat it.
That's because the certain benefit of a delicious dinner isn't worth a 20% risk of dying from it.
In other words, the effect of the *uncertain risk* by far outweighs the benefit.
Let's look at this in math. Basically, we need to compute the expected risk that we incur, i.e. we need to multiply the probability of the outcome with the benefit (or harm) associated with it:

$$L(\mathrm{action}| x) = \mathbf{E}_{y \sim p(y| x)}[\mathrm{loss}(\mathrm{action},y)]$$

Hence, the loss $L$ incurred by eating the mushroom is $L(a=\mathrm{eat}| x) = 0.2 * \infty + 0.8 * 0 = \infty$, whereas the cost of discarding it is $L(a=\mathrm{discard}| x) = 0.2 * 0 + 0.8 * 1 = 0.8$.

Our caution was justified: as any mycologist would tell us, the above mushroom actually *is* a death cap.
Classification can get much more complicated than just binary, multiclass, or even multi-label classification.
For instance, there are some variants of classification for addressing hierarchies.
Hierarchies assume that there exist some relationships among the many classes.
So not all errors are equal - we prefer to misclassify to a related class than to a distant class.
Usually, this is referred to as *hierarchical classification*.
One early example is due to [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus),
who organized the animals in a hierarchy.

![Classify sharks](../img/sharks.png)
:width:`500px`

In the case of animal classification, it might not be so bad to mistake a poodle for a schnauzer,
but our model would pay a huge penalty if it confused a poodle for a dinosaur.
Which hierarchy is relevant might depend on how you plan to use the model.
For example, rattle snakes and garter snakes might be close on the phylogenetic tree,
but mistaking a rattler for a garter could be deadly.

#### Tagging

Some classification problems don't fit neatly into the binary or multiclass classification setups.
For example, we could train a normal binary classifier to distinguish cats from dogs.
Given the current state of computer vision,
we can do this easily, with off-the-shelf tools.
Nonetheless, no matter how accurate our model gets, we might find ourselves in trouble when the classifier encounters an image of the Town Musicians of Bremen.

![A cat, a roster, a dog and a donkey](../img/stackedanimals.jpg)
:width:`500px`


As you can see, there's a cat in the picture, and a rooster, a dog, a donkey and a bird, with some trees in the background.
Depending on what we want to do with our model ultimately,
treating this as a binary classification problem
might not make a lot of sense.
Instead, we might want to give the model the option
of saying the image depicts a cat *and* a dog *and* a donkey *and* a rooster *and* a bird.

The problem of learning to predict classes
that are *not mutually exclusive*
is called multi-label classification.
Auto-tagging problems are typically best described
as multi-label classification problems.
Think of the tags people might apply to posts on a tech blog,
e.g., 'machine learning', 'technology', 'gadgets',
'programming languages', 'linux', 'cloud computing', 'AWS'.
A typical article might have 5-10 tags applied
because these concepts are correlated.
Posts about 'cloud computing' are likely to mention 'AWS'
and posts about 'machine learning' could also deal with 'programming languages'.

We also have to deal with this kind of problem when dealing with the biomedical literature,
where correctly tagging articles is important
because it allows researchers to do exhaustive reviews of the literature.
At the National Library of Medicine, a number of professional annotators
go over each article that gets indexed in PubMed
to associate it with the relevant terms from MeSH,
a collection of roughly 28k tags.
This is a time-consuming process and the annotators typically have a one year lag between archiving and tagging. Machine learning can be used here to provide provisional tags
until each article can have a proper manual review.
Indeed, for several years, the BioASQ organization has [hosted a competition](http://bioasq.org/)
to do precisely this.


#### Search and ranking

Sometimes we don't just want to assign each example to a bucket or to a real value. In the field of information retrieval, we want to impose a ranking on a set of items. Take web search for example, the goal is less to determine whether a particular page is relevant for a query, but rather, which one of the plethora of search results should be displayed for the user. We really care about the ordering of the relevant search results and our learning algorithm needs to produce ordered subsets of elements from a larger set. In other words, if we are asked to produce the first 5 letters from the alphabet, there is a difference between returning ``A B C D E`` and ``C A B E D``. Even if the result set is the same, the ordering within the set matters nonetheless.

One possible solution to this problem is to score every element in the set of possible sets along with a corresponding relevance score and then to retrieve the top-rated elements. [PageRank](https://en.wikipedia.org/wiki/PageRank) is an early example of such a relevance score. One of the peculiarities is that it didn't depend on the actual query. Instead, it simply helped to order the results that contained the query terms. Nowadays search engines use machine learning and behavioral models to obtain query-dependent relevance scores. There are entire conferences devoted to this subject.

<!-- Add / clean up-->

#### Recommender systems

Recommender systems are another problem setting that is related to search and ranking. The problems are  similar insofar as the goal is to display a set of relevant items to the user. The main difference is the emphasis on *personalization* to specific users in the context of recommender systems. For instance, for movie recommendations, the results page for a SciFi fan and the results page for a connoisseur of Woody Allen comedies might differ significantly.

Such problems occur, e.g. for movie, product or music recommendation. In some cases, customers will provide explicit details about how much they liked the product (e.g. Amazon product reviews). In some other cases, they might simply provide feedback if they are dissatisfied with the result (skipping titles on a playlist). Generally, such systems strive to estimate some score $y_{ij}$, such as an estimated rating or probability of purchase, given a user $u_i$ and product $p_j$.

Given such a model, then for any given user, we could retrieve the set of objects with the largest scores $y_{ij}$, which are then used as a recommendation. Production systems are considerably more advanced and take detailed user activity and item characteristics into account when computing such scores. The following image is an example of deep learning books recommended by Amazon based on personalization algorithms tuned to the author's preferences.

![Deep learning books recommended by Amazon.](../img/deeplearning_amazon.png)


#### Sequence Learning

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

These problems are among the most exciting applications of machine learning
and they are instances of *sequence learning*.
They require a model to either ingest sequences of inputs
or to emit sequences of outputs (or both!).
These latter problems are sometimes referred to as ``seq2seq`` problems.
Language translation is a ``seq2seq`` problem.
Transcribing text from spoken speech is also a ``seq2seq`` problem.
While it is impossible to consider all types of sequence transformations,
a number of special cases are worth mentioning:

##### Tagging and Parsing

This involves annotating a text sequence with attributes. In other words, the number of inputs and outputs is essentially the same. For instance, we might want to know where the verbs and subjects are. Alternatively, we might want to know which words are the named entities. In general, the goal is to decompose and annotate text based on structural and grammatical assumptions to get some annotation. This sounds more complex than it actually is. Below is a very simple example of annotating a sentence with tags indicating which words refer to named entities.

```text
Tom has dinner in Washington with Sally.
Ent  -    -    -     Ent      -    Ent
```


##### Automatic Speech Recognition

With speech recognition, the input sequence $x$ is the sound of a speaker,
and the output $y$ is the textual transcript of what the speaker said.
The challenge is that there are many more audio frames (sound is typically sampled at 8kHz or 16kHz) than text, i.e. there is no 1:1 correspondence between audio and text,
since thousands of samples correspond to a single spoken word.
These are ``seq2seq`` problems where the output is much shorter than the input.

![`-D-e-e-p- L-ea-r-ni-ng-`](../img/speech.png)
:width:`700px`

##### Text to Speech

Text-to-Speech (TTS) is the inverse of speech recognition.
In other words, the input $x$ is text
and the output $y$ is an audio file.
In this case, the output is *much longer* than the input.
While it is easy for *humans* to recognize a bad audio file,
this isn't quite so trivial for computers.

##### Machine Translation

Unlike the case of speech recognition, where corresponding inputs and outputs occur in the same order (after alignment),
in machine translation, order inversion can be vital.
In other words, while we are still converting one sequence into another,
neither the number of inputs and outputs
nor the order of corresponding data points
are assumed to be the same.
Consider the following illustrative example of the obnoxious tendency of Germans
<!-- Alex writing here -->
to place the verbs at the end of sentences.

```text
German:           Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
English:          Did you already check out this excellent tutorial?
Wrong alignment:  Did you yourself already this excellent tutorial looked-at?
```

A number of related problems exist.
For instance, determining the order in which a user reads a webpage
is a two-dimensional layout analysis problem.
Likewise, for dialogue problems,
we need to take world-knowledge and prior state into account.
This is an active area of research.


### Unsupervised learning

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
However, if you plan to be a data scientist, you'd better get used to it.
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


### Interacting with an Environment

So far, we haven't discussed where data actually comes from,
or what actually *happens* when a machine learning model generates an output.
That's because supervised learning and unsupervised learning
do not address these issues in a very sophisticated way.
In either case, we grab a big pile of data up front,
then do our pattern recognition without ever interacting with the environment again.
Because all of the learning takes place after the algorithm is disconnected from the environment,
this is called *offline learning*.
For supervised learning, the process looks like this:

![Collect data for supervised learning from an environment.](../img/data-collection.svg)


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


Considering the interaction with an environment opens a whole set of new modeling questions. Does the environment:

* remember what we did previously?
* want to help us, e.g. a user reading text into a speech recognizer?
* want to beat us, i.e. an adversarial setting like spam filtering (against spammers) or playing a game (vs an opponent)?
* not  care (as in most cases)?
* have shifting dynamics (steady vs. shifting over time)?

This last question raises the problem of *covariate shift*,
(when training and test data are different).
It's a problem that most of us have experienced when taking exams written by a lecturer,
while the homeworks were composed by his TAs.
We'll briefly describe reinforcement learning and adversarial learning,
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
The agent then receives a subsequent observation, and chooses a subsequent action, and so on.
The behavior of an RL agent is governed by a *policy*.
In short, a *policy* is just a function that maps from observations (of the environment) to actions.
The goal of reinforcement learning is to produce a good policy.

![The interaction between reinforcement learning and an environment.](../img/rl-environment.svg)

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
might require considering its previous observations before entering the closet.

Finally, at any given point, reinforcement learners might know of one good policy,
but there might be many other better policies that the agent has never tried.
The reinforcement learner must constantly choose
whether to *exploit* the best currently-known strategy as a policy,
or to *explore* the space of strategies,
potentially giving up some short-run reward in exchange for knowledge.


#### MDPs, bandits, and friends

The general reinforcement learning problem
is a very general setting.
Actions affect subsequent observations.
Rewards are only observed corresponding to the chosen actions.
The environment may be either fully or partially observed.
Accounting for all this complexity at once may ask too much of researchers.
Moreover not every practical problem exhibits all this complexity.
As a result, researchers have studied a number of *special cases* of reinforcement learning problems.

When the environment is fully observed,
we call the RL problem a *Markov Decision Process* (MDP).
When the state does not depend on the previous actions,
we call the problem a *contextual bandit problem*.
When there is no state, just a set of available actions with initially unknown rewards,
this problem is the classic *multi-armed bandit problem*.



## Roots

Although deep learning is a recent invention, humans have held the desire to analyze data and to predict future outcomes for centuries. In fact, much of natural science has its roots in this. For instance, the Bernoulli distribution is named after [Jacob Bernoulli (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli), and the Gaussian distribution was discovered by [Carl Friedrich Gauss (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss). He invented for instance the least mean squares algorithm, which is still used today for a range of problems from insurance calculations to medical diagnostics. These tools gave rise to an experimental approach in natural sciences - for instance, Ohm's law relating current and voltage in a resistor is perfectly described by a linear model.

Even in the middle ages mathematicians had a keen intuition of estimates. For instance, the geometry book of [Jacob Köbel (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry) illustrates averaging the length of 16 adult men's feet to obtain the average foot length.

![Estimating the length of a foot](../img/koebel.jpg)
:width:`500px`

Figure 1.1 illustrates how this estimator works. 16 adult men were asked to line up in a row, when leaving church. Their aggregate length was then divided by 16 to obtain an estimate for what now amounts to 1 foot. This 'algorithm' was later improved to deal with misshapen feet - the 2 men with the shortest and longest feet respectively were sent away, averaging only over the remainder. This is one of the earliest examples of the trimmed mean estimate.

Statistics really took off with the collection and availability of data. One of its titans, [Ronald Fisher (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher), contributed significantly to its theory and also its applications in genetics. Many of his algorithms (such as Linear Discriminant Analysis) and formulae (such as the Fisher Information Matrix) are still in frequent use today (even the Iris dataset that he released in 1936 is still used sometimes to illustrate machine learning algorithms).

A second influence for machine learning came from Information Theory [(Claude Shannon, 1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon) and the Theory of computation via [Alan Turing (1912-1954)](https://en.wikipedia.org/wiki/Alan_Turing). Turing posed the question "can machines think?” in his famous paper [Computing machinery and intelligence](https://en.wikipedia.org/wiki/Computing_Machinery_and_Intelligence) (Mind, October 1950). In what he described as the Turing test, a machine can be considered intelligent if it is difficult for a human evaluator to distinguish between the replies from a machine and a human being through textual interactions. To this day, the development of intelligent machines is changing rapidly and continuously.

Another influence can be found in neuroscience and psychology. After all, humans clearly exhibit intelligent behavior. It is thus only reasonable to ask whether one could explain and possibly reverse engineer these insights. One of the oldest algorithms to accomplish this was formulated by [Donald Hebb (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb).

In his groundbreaking book [The Organization of Behavior](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf) (John Wiley & Sons, 1949) he posited that neurons learn by positive reinforcement. This became known as the Hebbian learning rule. It is the prototype of Rosenblatt's perceptron learning algorithm and it laid the foundations of many stochastic gradient descent algorithms that underpin deep learning today: reinforce desirable behavior and diminish undesirable behavior to obtain good weights in a neural network.

Biological inspiration is what gave Neural Networks its name. For over a century (dating back to the models of Alexander Bain, 1873 and James Sherrington, 1890) researchers have tried to assemble computational circuits that resemble networks of interacting neurons. Over time the interpretation of biology became more loose but the name stuck. At its heart lie a few key principles that can be found in most networks today:

* The alternation of linear and nonlinear processing units, often referred to as 'layers'.
* The use of the chain rule (aka backpropagation) for adjusting parameters in the entire network at once.

After initial rapid progress, research in Neural Networks languished from around 1995 until 2005. This was due to a number of reasons. Training a network is computationally very expensive. While RAM was plentiful at the end of the past century, computational power was scarce. Secondly, datasets were relatively small. In fact, Fisher's 'Iris dataset' from 1932 was a popular tool for testing the efficacy of algorithms. MNIST with its 60,000 handwritten digits was considered huge.

Given the scarcity of data and computation, strong statistical tools such as Kernel Methods, Decision Trees and Graphical Models proved empirically superior. Unlike Neural Networks they did not require weeks to train and provided predictable results with strong theoretical guarantees.

## The Road to Deep Learning

Much of this changed with the ready availability of large amounts of data, due to the World Wide Web, the advent of companies serving hundreds of millions of users online, a dissemination of cheap, high quality sensors, cheap data storage (Kryder's law), and cheap computation (Moore's law), in particular in the form of GPUs, originally engineered for computer gaming. Suddenly algorithms and models that seemed computationally infeasible became relevant (and vice versa). This is best illustrated in :numref:`tab_intro_decade`.

:Dataset versus computer memory and computational power

|Decade|Dataset|Memory|Floating Point Calculations per Second|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (Nvidia C2050)|
|2020|1 T (social network)|100 GB|1 PF (Nvidia DGX-2)|
:label:`tab_intro_decade`

It is quite evident that RAM has not kept pace with the growth in data. At the same time, the increase in computational power has outpaced that of the data available. This means that statistical models needed to become more memory efficient (this is typically achieved by adding nonlinearities) while simultaneously being able to spend more time on optimizing these parameters, due to an increased compute budget. Consequently the sweet spot in machine learning and statistics moved from (generalized) linear models and kernel methods to deep networks. This is also one of the reasons why many of the mainstays of deep learning, such as Multilayer Perceptrons (e.g. McCulloch & Pitts, 1943), Convolutional Neural Networks (Le Cun, 1992), Long Short Term Memory (Hochreiter & Schmidhuber, 1997), Q-Learning (Watkins, 1989), were essentially 'rediscovered' in the past decade, after laying dormant for considerable time.

The recent progress in statistical models, applications, and algorithms, has sometimes been likened to the Cambrian Explosion: a moment of rapid progress in the evolution of species. Indeed, the state of the art is not just a mere consequence of available resources, applied to decades old algorithms. Note that the list below barely scratches the surface of the ideas that have helped researchers achieve tremendous progress over the past decade.

* Novel methods for capacity control, such as Dropout [3] allowed for training of relatively large networks without the danger of overfitting, i.e. without the danger of merely memorizing large parts of the training data. This was achieved by applying noise injection [4] throughout the network, replacing weights by random variables for training purposes.
* Attention mechanisms solved a second problem that had plagued statistics for over a century: how to increase the memory and complexity of a system without increasing the number of learnable parameters. [5] found an elegant solution by using what can only be viewed as a learnable pointer structure. That is, rather than having to remember an entire sentence, e.g. for machine translation in a fixed-dimensional representation, all that needed to be stored was a pointer to the intermediate state of the translation process. This allowed for significantly increased accuracy for long sentences, since the model no longer needed to remember the entire sentence before beginning to generate sentences.
* Multi-stage designs, e.g. via the Memory Networks [6] and the Neural Programmer-Interpreter [7] allowed statistical modelers to describe iterative approaches to reasoning. These tools allow for an internal state of the deep network to be modified repeatedly, thus carrying out subsequent steps in a chain of reasoning, similar to how a processor can modify memory for a computation.
* Another key development was the invention of Generative Adversarial Networks [8]. Traditionally statistical methods for density estimation and generative models focused on finding proper probability distributions and (often approximate) algorithms for sampling from them. As a result, these algorithms were largely limited by the lack of flexibility inherent in the statistical models. The crucial innovation in GANs was to replace the sampler by an arbitrary algorithm with differentiable parameters. These are then adjusted in such a way that the discriminator (effectively a two-sample test) cannot distinguish fake from real data. Through the ability to use arbitrary algorithms to generate data it opened up density estimation to a wide variety of techniques. Examples of galloping Zebras [9] and of fake celebrity faces [10] are both testimony to this progress.
* In many cases a single GPU is insufficient to process the large amounts of data available for training. Over the past decade the ability to build parallel distributed training algorithms has improved significantly. One of the key challenges in designing scalable algorithms is that the workhorse of deep learning optimization, stochastic gradient descent, relies on relatively small minibatches of data to be processed. At the same time, small batches limit the efficiency of GPUs. Hence, training on 1024 GPUs with a minibatch size of, say 32 images per batch amounts to an aggregate minibatch of 32k images. Recent work, first by Li [11],  and subsequently by You et al. [12] and Jia et al. [13] pushed the size up to 64k observations, reducing training time for ResNet50 on ImageNet to less than 7 minutes. For comparison - initially training times were measured in the order of days.
* The ability to parallelize computation has also contributed quite crucially to progress in reinforcement learning, at least whenever simulation is an option. This has led to significant progress in computers achieving superhuman performance in Go, Atari games, Starcraft, and in physics simulations (e.g. using MuJoCo). See e.g. Silver et al. [18] for a description of how to achieve this in AlphaGo. In a nutshell, reinforcement learning works best if plenty of (state, action, reward) triples are available, i.e. whenever it is possible to try out lots of things to learn how they relate to each other. Simulation provides such an avenue.
* Deep Learning frameworks have played a crucial role in disseminating ideas. The first generation of frameworks allowing for easy modeling encompassed [Caffe](https://github.com/BVLC/caffe), [Torch](https://github.com/torch), and [Theano](https://github.com/Theano/Theano). Many seminal papers were written using these tools. By now they have been superseded by [TensorFlow](https://github.com/tensorflow/tensorflow), often used via its high level API [Keras](https://github.com/keras-team/keras), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2), and [Apache MxNet](https://github.com/apache/incubator-mxnet). The third generation of tools, namely imperative tools for deep learning, was arguably spearheaded by [Chainer](https://github.com/chainer/chainer), which used a syntax similar to Python NumPy to describe models. This idea was adopted by [PyTorch](https://github.com/pytorch/pytorch) and the [Gluon API](https://github.com/apache/incubator-mxnet) of MxNet. It is the latter that this course uses to teach Deep Learning.

The division of labor between systems researchers building better tools for training and statistical modelers building better networks has greatly simplified things. For instance, training a linear logistic regression model used to be a nontrivial homework problem, worthy to give to new Machine Learning PhD students at Carnegie Mellon University in 2014. By now, this task can be accomplished with less than 10 lines of code, putting it firmly into the grasp of programmers.

## Success Stories

Artificial Intelligence has a long history of delivering results that would be difficult to accomplish otherwise. For instance, mail is sorted using optical character recognition. These systems have been deployed since the 90s (this is, after all, the source of the famous MNIST and USPS sets of handwritten digits). The same applies to reading checks for bank deposits and scoring creditworthiness of applicants. Financial transactions are checked for fraud automatically. This forms the backbone of many e-commerce payment systems, such as PayPal, Stripe, AliPay, WeChat, Apple, Visa, MasterCard. Computer programs for chess have been competitive for decades. Machine learning feeds search, recommendation, personalization and ranking on the internet. In other words, artificial intelligence and machine learning are pervasive, albeit often hidden from sight.

It is only recently that AI has been in the limelight, mostly due to solutions to problems that were considered intractable previously.

* Intelligent assistants, such as Apple's Siri, Amazon's Alexa, or Google's assistant are able to answer spoken questions with a reasonable degree of accuracy. This includes menial tasks such as turning on light switches (a boon to the disabled) up to making barber's appointments and offering phone support dialog. This is likely the most noticeable sign that AI is affecting our lives.

* A key ingredient in digital assistants is the ability to recognize speech accurately. Gradually the accuracy of such systems has increased to the point where they reach human parity [14] for certain applications.
* Object recognition likewise has come a long way. Estimating the object in a picture was a fairly challenging task in 2010. On the ImageNet benchmark Lin et al. [15] achieved a top-5 error rate of 28%. By 2017 Hu et al. [16] reduced this error rate to 2.25%. Similarly stunning results have been achieved for identifying birds, or diagnosing skin cancer.
* Games used to be a bastion of human intelligence. Starting from TDGammon [23], a program for playing Backgammon using temporal difference (TD) reinforcement learning, algorithmic and computational progress has led to algorithms for a wide range of applications. Unlike Backgammon, chess has a much more complex state space and set of actions. DeepBlue beat Gary Kasparov, Campbell et al. [17], using massive parallelism, special purpose hardware and efficient search through the game tree. Go is more difficult still, due to its huge state space. AlphaGo reached human parity in 2015,  Silver et al. [18] using Deep Learning combined with Monte Carlo tree sampling. The challenge in Poker was that the state space is large and it is not fully observed (we don't know the opponents' cards). Libratus exceeded human performance in Poker using efficiently structured strategies; Brown and Sandholm [19]. This illustrates the impressive progress in games and the fact that advanced algorithms played a crucial part in them.
* Another indication of progress in AI is the advent of self-driving cars and
  trucks. While full autonomy is not quite within reach yet, excellent progress
  has been made in this direction, with companies such as
  Momenta, Tesla,
  NVIDIA, MobilEye
  and Waymo shipping products that enable at least
  partial autonomy. What makes full autonomy so challenging is that proper
  driving requires the ability to perceive, to reason and to incorporate rules
  into a system. At present, Deep Learning is used primarily in the computer
  vision aspect of these problems. The rest is heavily tuned by engineers.

Again, the above list barely scratches the surface of what is considered intelligent and where machine learning has led to impressive progress in a field. For instance, robotics, logistics, computational biology, particle physics and astronomy owe some of their most impressive recent advances at least in parts to machine learning. ML is thus becoming a ubiquitous tool for engineers and scientists.

Frequently the question of the AI apocalypse, or the AI singularity has been raised in non-technical articles on AI. The fear is that somehow machine learning systems will become sentient and decide independently from their programmers (and masters) about things that directly affect the livelihood of humans. To some extent AI already affects the livelihood of humans in an immediate way - creditworthiness is assessed automatically, autopilots mostly navigate cars safely, decisions about whether to grant bail use statistical data as input. More frivolously, we can ask Alexa to switch on the coffee machine and she will happily oblige, provided that the appliance is internet enabled.

Fortunately we are far from a sentient AI system that is ready to enslave its human creators (or burn their coffee). Firstly, AI systems are engineered, trained and deployed in a specific, goal oriented manner. While their behavior might give the illusion of general intelligence, it is a combination of rules, heuristics and statistical models that underlie the design. Second, at present tools for general Artificial Intelligence simply do not exist that are able to improve themselves, reason about themselves, and that are able to modify, extend and improve their own architecture while trying to solve general tasks.

A much more realistic concern is how AI is being used in our daily lives. It is likely that many menial tasks fulfilled by truck drivers and shop assistants can and will be automated. Farm robots will likely reduce the cost for organic farming but they will also automate harvesting operations. This phase of the industrial revolution will have profound consequences on large swaths of society (truck drivers and shop assistants are some of the most common jobs in many states). Furthermore, statistical models, when applied without care can lead to racial, gender or age bias. It is important to ensure that these algorithms are used with great care. This is a much bigger concern than to worry about a potentially malevolent superintelligence intent on destroying humanity.

## Summary

* Machine learning studies how computer systems can use data to improve performance. It combines ideas from statistics, data mining, artificial intelligence and optimization. Often it is used as a means of implementing artificially intelligent solutions.
* As a class of machine learning, representational learning focuses on how to automatically find the appropriate way to represent data. This is often accomplished by a progression of learned transformations.
* Much of the recent progress has been triggered by an abundance of data arising from cheap sensors and internet scale applications, and by significant progress in computation, mostly through GPUs.
* Whole system optimization is a key component in obtaining good performance. The availability of efficient deep learning frameworks has made design and implementation of this significantly easier.

## Exercises

1. Which parts of code that you are currently writing could be 'learned', i.e. improved by learning and automatically determining design choices that are made in your code? Does your code include heuristic design choices?
1. Which problems that you encounter have many examples for how to solve them, yet no specific way to automate them? These may be prime candidates for using Deep Learning.
1. Viewing the development of Artificial Intelligence as a new industrial revolution, what is the relationship between algorithms and data? Is it similar to steam engines and coal (what is the fundamental difference)?
1. Where else can you apply the end-to-end training approach? Physics? Engineering? Econometrics?

## References

[1] Turing, A. M. (1950). Computing machinery and intelligence. Mind, 59(236), 433.

[2] Hebb, D. O. (1949). The organization of behavior; a neuropsychological theory. A Wiley Book in Clinical Psychology. 62-78.

[3] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.

[4] Bishop, C. M. (1995). Training with noise is equivalent to Tikhonov regularization. Neural computation, 7(1), 108-116.

[5] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

[6] Sukhbaatar, S., Weston, J., & Fergus, R. (2015). End-to-end memory networks. In Advances in neural information processing systems (pp. 2440-2448).

[7] Reed, S., & De Freitas, N. (2015). Neural programmer-interpreters. arXiv preprint arXiv:1511.06279.

[8] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

[9] Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint.

[10] Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017). Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196.

[11] Li, M. (2017). Scaling Distributed Machine Learning with System and Algorithm Co-design (Doctoral dissertation, PhD thesis, Intel).

[12] You, Y., Gitman, I., & Ginsburg, B. Large batch training of convolutional networks. ArXiv e-prints.

[13] Jia, X., Song, S., He, W., Wang, Y., Rong, H., Zhou, F., … & Chen, T. (2018). Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes. arXiv preprint arXiv:1807.11205.

[14] Xiong, W., Droppo, J., Huang, X., Seide, F., Seltzer, M., Stolcke, A., … & Zweig, G. (2017, March). The Microsoft 2016 conversational speech recognition system. In Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on (pp. 5255-5259). IEEE.

[15] Lin, Y., Lv, F., Zhu, S., Yang, M., Cour, T., Yu, K., … & Huang, T. (2010). Imagenet classification: fast descriptor coding and large-scale svm training. Large scale visual recognition challenge.

[16] Hu, J., Shen, L., & Sun, G. (2017). Squeeze-and-excitation networks. arXiv preprint arXiv:1709.01507, 7.

[17] Campbell, M., Hoane Jr, A. J., & Hsu, F. H. (2002). Deep blue. Artificial intelligence, 134 (1-2), 57-83.

[18] Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., … & Dieleman, S. (2016). Mastering the game of Go with deep neural networks and tree search. Nature, 529 (7587), 484.

[19] Brown, N., & Sandholm, T. (2017, August). Libratus: The superhuman ai for no-limit poker. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence.

[20] Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6), 679-698.

[21] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2), 91-110.

[22] Salton, G., & McGill, M. J. (1986). Introduction to modern information retrieval.

[23] Tesauro, G. (1995), Transactions of the ACM, (38) 3, 58-68


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2310)

![](../img/qr_intro.svg)
