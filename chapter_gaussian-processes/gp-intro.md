# Introduction to Gaussian Processes

In many cases, machine learning amounts to estimating parameters from data. These parameters are often numerous and  relatively uninterpretable --- such as the weights of a neural network. Gaussian processes, by contrast, provide a mechanism for directly reasoning about the high-level properties of functions that could fit our data. For example, we may have a sense of whether the functions that fit our data are quickly varying, periodic, involve conditional independencies, or translation invariance. Gaussian processes enable us to easily incorporate these properties into our model, by directly specifying a Gaussian distribution over the function values that could fit our data. 

Let's get a feel for how Gaussian processes working with some examples.

Suppose we observe the following dataset, of regression targets (outputs), y, indexed by inputs, x. As an example, the targets could be changes in carbon dioxide concentrations, and the inputs could be the times at which these targets have been recorded. What are some features of the data? How quickly does it seem to varying? Do we have data points collected at regular intervals, or are there missing inputs? How would you imagine filling in the missing regions, or forecasting up until x=20?

![data](https://user-images.githubusercontent.com/6753639/178247765-650772fb-2622-42d0-8eff-316dc835816f.png)

In order to fit the data with a Gaussian process, we start by specifying a prior distribution over what types of functions we might believe to be reasonable. Here we show several sample functions from a Gaussian process. Does this prior look reasonable? Note here we are not looking for functions that fit our dataset, but instead for specifying reasonable high-level properties of the solutions, such as how quickly they vary with inputs. 

![priorsamp](https://user-images.githubusercontent.com/6753639/178247905-ca6d5812-92eb-45d2-9004-a435da917e78.png)

Once we condition on data, we can use this prior to infer a posterior distribution over functions that could fit the data. Here we show sample posterior functions.

![postsampnomean](https://user-images.githubusercontent.com/6753639/178248696-bb31053e-68c9-4679-b09b-59a319d6479b.png)

We see that each of these functions are entirely consistent with our data, perfectly running through each observation. In order to use these posterior samples to make predictions, we can average the values of every possible sample function from the posterior, to create the curve below, in thick blue. Note that we don't actually have to take an infinite number of samples to compute this expection; as we will see later, we can compute the expectation in closed form.  

![postsamp](https://user-images.githubusercontent.com/6753639/178248173-9d13e613-85f3-4414-ab13-8eb763580225.png)

We may also want a representation of uncertainty, so we know how confident we should be in our predictions. Intuitively, we should have more uncertainty where there is more variability in the sample posterior functions, as this tells us there are many more possible values the true function could take. This type of uncertainty is called _epistemic uncertainty_, which is the _reducible uncertainty_ associated with lack of information. As we acquire more data, this type of uncertainty disappears, as there will be increasingly fewer solutions consistent with what we observe. Like with the posterior mean, we can compute the posterior variance (the variability of these functions in the posterior) in closed form. With shade, we show two times the posterior standard deviation on either side of the mean, creating a _credible interval_ that has a 95% probability of containing the true value of the function for any input $x$.

![postsampcredibleset](https://user-images.githubusercontent.com/6753639/178248952-b14e3d72-e65f-41ed-9577-8d3363c8cf11.png)

The plot looks somewhat cleaner if we remove the posterior samples, simply visualizing the data, posterior mean, and 95% credible set. Notice how the uncertainty grows away from the data, a train of epistemic uncertainty. 

![datafitcredibleset](https://user-images.githubusercontent.com/6753639/178249137-23af70e9-0753-4491-9215-7a757ff60652.png)





