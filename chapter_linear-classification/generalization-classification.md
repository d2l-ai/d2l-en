# Generalization in Classification

:label:`chap_classification_generalization`

In 

[TODO CLEAN UP]

[GOALS]

Add bits about estimating the error rate using holdout data,
how fast the estimate converges, aysmptotic binomial confidence interval, finite sample guarantee via Hoeffdings 





[OLD BITS MOVED OVER FROM LINEAR]

Fortunately, our specific choice of loss function ensures that minimizing it will also lead to maximum accuracy. This is the case since the maximum likelihood estimator is consistent. It follows as a special case of the Cramer-Rao bound :cite:`cramer1946mathematical,radhakrishna1945information`. For more work on consistency see also :cite:`zhang2004statistical`.




loss function that we are actually minimizing. Fortunately, our specific choice of loss function ensures that minimizing it will also lead to maximum accuracy. This is the case since the maximum likelihood estimator is consistent. It follows as a special case of the Cramer-Rao bound :cite:`cramer1946mathematical,radhakrishna1945information`. For more work on consistency see also :cite:`zhang2004statistical`.

More generally, though, the decision of which category to pick is far from trivial. For instance, when deciding where to assign an e-mail to, mistaking a "Primary" e-mail for a "Social" e-mail might be undesirable but far less disastrous than moving it to the spam folder (and later automatically deleting it). As such, we will tend to err on the side of caution with regard to assigning any e-mail to the "Spam" folder, rather than picking the most likely category.





%
% Move to classification generalization
% 


### Statistical Learning Theory

Because generalization is such a fundamental problem,
generations of theorists have dedicated careers 
to developing formal theories 
that explain when we should expect models to generalize.
In their [epoynmous theorem](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem) 
Glivenko and Cantelli derived the rate
at which the training error converges 
to the generalization error :cite:`Glivenko33,Cantelli33`. 
In a series of seminal papers, Vapnik and Chervonenkis
extended this theory to more general classes of functions 
:cite:`VapChe64,VapChe68,VapChe71,VapChe74b,VapChe81,VapChe91`.
However 




%
%
%



Being a good machine learning scientist requires thinking critically,
and already you should be poking holes in this assumption,
coming up with common cases where the assumption fails.
What if we train a mortality risk predictor
on data collected from patients at UCSF Medical Center,
and apply it on patients at Massachusetts General Hospital?
These distributions are simply not identical. This is a well-studied 
problem in statistics :cite:`rosenbaum1983central`.
Moreover, draws might be correlated in time.
What if we are classifying the topics of Tweets?
The news cycle would create temporal dependencies
in the topics being discussed, violating any assumptions of independence. 

Sometimes we can get away with minor violations of the IID assumption
and our models will continue to work remarkably well.
After all, nearly every real-world application
involves at least some minor violation of the IID assumption,
and yet we have many useful tools for
various applications such as
face recognition,
speech recognition, and language translation. :cite:`yu1994rates` provides 
a quantitative handle on this behavior. 

Other violations are sure to cause trouble.
Imagine, for example, if we try to train
a face recognition system by training it
exclusively on university students
and then want to deploy it as a tool
for monitoring geriatrics in a nursing home population.
This is unlikely to work well since college students
tend to look considerably different from the elderly.

In subsequent chapters, we will discuss problems
arising from violations of the IID assumption.
For now, even taking the IID assumption for granted,
understanding generalization is a formidable problem.
Moreover, elucidating the precise theoretical foundations
that might explain why deep neural networks generalize as well as they do
continues to vex the greatest minds in learning theory :cite:`frankle2018lottery,bartlett2021deep,nagarajan2019uniform,kawaguchi2017generalization`.



When we train our models, we attempt to search for a function
that fits the training data as well as possible.
If the function is so flexible that it can catch on to spurious patterns
just as easily as to true associations,
then it might perform *too well* without producing a model
that generalizes well to unseen data.
This is precisely what we want to avoid or at least control.
Many of the techniques in deep learning are heuristics and tricks
aimed at guarding against overfitting (:numref:`sec_weight_decay`, :numref:`sec_dropout`, :numref:`sec_batch_norm`).







Often with neural networks, we think of a model
that takes more training iterations as more complex,
and one subject to *early stopping* (fewer training iterations) as less complex :cite:`prechelt1998early`.



Much of the intuition of this arises from Statistical Learning Theory. 
One of the guarantees it provides :cite:`Vapnik98` 
is that the gap between empirical risk and expected risk is bounded by 

$$\Pr\left(R[p, f] - R_\mathrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \epsilon\right) \geq 1-\delta
\ \text{for}\ \epsilon \geq c \sqrt{(\mathrm{VC} - \log \delta)/n}.$$

Here $\delta > 0$ is the probability that the bound is violated 
and $\mathrm{VC}$ is the Vapnik-Chervonenkis (VC)
dimension of the set of functions that we want to fit. 
For instance, for polynomials of degree $d$ the VC dimension is $d+1$. 
Lastly, $c > 0$ is a constant that depends 
only on the scale of the loss that can be incurred. 
In short, this shows that our bound becomes increasingly loose 
as we pick more complex models 
and that the number of free parameters 
should not increase more rapidly 
than the dataset size $n$ increases. 
See :cite:`boucheron2005theory` for a detailed discussion,
including several advanced ways of measuring function complexity.
