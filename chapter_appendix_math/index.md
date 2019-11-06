# Appendix: Mathematics for Deep Learning
:label:`chap_appendix_math`

**Brent Werness** (*Amazon*) and **Rachel Hu** (*Amazon*)


One of the wonderful parts of modern deep learning is the fact that much of it can be understood and used without a full understanding of the mathematics below it.  This is a sign of the fact that the field is becoming more mature.  Most software developers no longer need to worry about the theory of computable functions, or if programming languages without a ```goto``` can emulate programming languages with a ```goto``` with at most constant overhead, and neither should the deep learning practitioner need to worry about the theoretical foundations maximum likelihood learning, if one can find an architecture to approximate a target function to an arbitrary degree of accuracy.

That said, we are not quite there yet.

Sometimes when building a model in practice you will need to understand how architectural choices influence gradient flow, or what assumptions you are making by training with a certain loss function.  You might need to know what in the world entropy measures, and how it can help you understand exactly what bits-per-character means in your model.  Once you get on that level of interaction, mathematical depth again becomes a must.

This appendix aims to provide you all the mathematical background you need to understand the core theory of modern deep learned systems.  This topic is unboundedly deep, so this is not exhaustive.

We will begin with examining linear algebra in greater depth.  Key to this chapter is developing the geometric understanding of all these components that will allow you to visualize what various operations are doing to your data.  Within this section, a development of the basics of eigen-decompositions is a must.

Next, we move on to a more complete understanding of calculus, single variable, multivariable, and integral.  This section develops the theory of calculus to the point that we can fully understand why the gradient is the direction of steepest descent, and why back-propagation takes the form it does.  Integral calculus is developed only to the degree needed to support our next topic, probability theory.

Probability theory is the mathematical theory of chance and uncertainty.  Problems encountered in practice frequently are not certain, and thus we need a language to speak about uncertain things.  We develop the theory of random variables to the point that we can discuss models probabilistically, and learn about the most commonly encountered distributions.  This allows us to then see an example of Naive Bayes, which is a probabilistic classification technique that requires nothing but the fundamentals of probability to understand fully.

Closely related to probability theory is the study of statistics.  In particular, the area of inferential statistics allows us to evaluate if results we observe to try and understand properties of the underlying distribution (for instance, are these two distributions the same or different).  We will examine the concepts of confidence intervals and hypothesis testing to provide the language needed to discuss similarities and differences in observed distributions.

Lastly, we turn to the topic of information theory, which is the mathematical study of information storage and transmission.  This provides the core language by which we may discuss quantitatively how much information a model holds on a domain of discourse.

Taken together, these form the core of the mathematical concepts needed to begin down the path towards a deep understanding of ML models.

```toc
:maxdepth: 2

linear-algebra
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
info-theory
```

