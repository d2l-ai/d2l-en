# Introduction

Until recently, nearly all of the computer programs
thay we interacted with every day were coded
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
And when you're able to devise solutions that work $100\%$ of the time.
*you should not be using machine learning*.

Fortunately—for the growing community of ML scientists—many
problems in automation don't bend so easily to human ingenuity.
Imagine huddling around the whiteboard with the smartest minds you know,
but this time you are tackling any of the following problems:
 * Write a program that predicts tomorrow's weather
given geographic information, satellite images,
and a trailing window of past weather.
 * Write a program that takes in a question,
 expressed in free-form text, and answers it correctly.
 * Write a program that given an image
 can identify all the people it contains,
 drawing outlines around each.
  * Write a progam that presents users with products
  that they are likely to enjoy but unlikely,
  in the natural course of browsing, to encounter.

In each of these cases, even the best minds
are incapable of tackling the problems from scratch.
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




```eval_rst

.. toctree::
   :maxdepth: 2

   motivating-example
   basics-of-ml
   kinds-of-ml
   history
   summary
   problems
   discussion
   references
```
