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

The properties of the Gaussian process that we used to fit the data are strongly controlled by what's called a _covariance function_, also known as a _kernel_. The covariance function we used is called the _RBF (Radial Basis Function) kernel_, which has the form
$$ k_{\text{RBF}}(x,x') = \text{cov}(f(x),f(x')) = a^2 \exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right) $$

The _hyperparameters_ of this kernel are interpretable. The _amplitude_ parameter $a$ controls the vertical scale over which the function is varying, and the _length-scale_ parameter $\ell$ controls the rate of variation (the wiggliness) of the function. Larger $a$ means larger function values, and larger $\ell$ means more slowly varying functions. Let's see what happens to our sample prior and posterior functions as we vary $a$ and $\ell$. 

The _length-scale_ has a particularly pronounced effect on the predictions and uncertainty of a GP. At $||x-x'|| = \ell$, the covariance between a pair of function values is $a^2\exp(-0.5)$. At larger distances than $\ell$, the values of the function values becomes nearly uncorrelated. This means that if we want to make a prediction at a point $x_*$, then function values with inputs $x$ such that $||x-x'||>\ell$ will not have a strong effect on our predictions. 

Let's see how changing the lengthscale affects sample prior and posterior functions, and credible sets. The above fits use a length-scale of $2$. Let's now consider $\ell = 0.1, 0.5, 2, 5, 10$. A length-scale of $0.1$ is very small relative to the range of the input domain we are considering, $25$. For example, the values of the function at $x=5$ and $x=10$ will have essentially no correlation at such a length-scale. On the other hand, for a length-scale of $10$, the function values at these inputs will be highly correlated. Note that the vertical scale changes in the following figures.

![priorpoint1](https://user-images.githubusercontent.com/6753639/178250594-d2032bcd-f5bc-4938-8cfa-aa1658c18425.png)
![postpoint1](https://user-images.githubusercontent.com/6753639/178250619-121ad67f-45f4-47ae-9637-c5f367afd211.png)

![priorpoint5](https://user-images.githubusercontent.com/6753639/178250705-1f0ec480-235e-4ad7-a3c6-a282d8d4e60b.png)
![postpoint5](https://user-images.githubusercontent.com/6753639/178250716-9238a419-e43e-405e-b1e3-857790ce52c3.png)

![prior2](https://user-images.githubusercontent.com/6753639/178250738-dd0708de-c008-4708-9a3c-5466b0ac6504.png)
![post2](https://user-images.githubusercontent.com/6753639/178250763-066698cc-4b93-496f-8a01-c2b1f1d6815c.png)

![prior5](https://user-images.githubusercontent.com/6753639/178250780-e5c522b7-f9c7-416c-8017-3cb921ff14b2.png)
![post5](https://user-images.githubusercontent.com/6753639/178250794-89470592-cdb3-4e63-b0d8-d66f002fc593.png)

![prior10](https://user-images.githubusercontent.com/6753639/178250805-080a5c66-69ec-456d-ade5-e0664874782f.png)
![post10](https://user-images.githubusercontent.com/6753639/178250815-588fccee-bfcd-4d46-87af-7429596ddc6e.png)

Notice as the length-scale increases the 'wiggliness' of the functions decrease, and our uncertainty decreases. If the length-scale is small, the uncertainty will quickly increase as we move away from the data, as the datapoints become less informative about teh function values. 

Now, let's vary the amplitude parameter, holding the length-scale fixed at $2$. Note the vertical scale is held fixed for the prior samples, and varies for the posterior samples, so you can clearly see both the increasing scale of the function, and the fits to the data.

![priorap1](https://user-images.githubusercontent.com/6753639/178252126-8a984a0c-56f8-409c-b817-68b21af98582.png)
![postapoint1](https://user-images.githubusercontent.com/6753639/178252136-868dd45a-b21e-4311-8164-a60ea41c221c.png)

![priora2](https://user-images.githubusercontent.com/6753639/178252163-c9ac2360-6bee-44fe-985c-731101d8c575.png)
![posta2](https://user-images.githubusercontent.com/6753639/178252195-c325e446-4c61-4851-a841-b547bbab2e2d.png)

![priora4](https://user-images.githubusercontent.com/6753639/178252214-eca6fe2a-0af0-4a13-a71f-4851c02dc4d7.png)
![posta4](https://user-images.githubusercontent.com/6753639/178252232-38e229a9-bf48-4a67-9883-3cf494f8ff6a.png)

![priora8](https://user-images.githubusercontent.com/6753639/178252271-ccabde74-8ec3-44d1-9842-6309444c4ab5.png)
![posta8](https://user-images.githubusercontent.com/6753639/178252284-b59daae3-2648-4ef6-bc09-7c0b4d9a4f02.png)

![priora16](https://user-images.githubusercontent.com/6753639/178252311-b9a5c51f-a0f8-4d65-ba5e-1f8b917c0d7c.png)
![posta16](https://user-images.githubusercontent.com/6753639/178252339-db99413b-78f7-41f7-8c6b-92fe10d634a4.png)










