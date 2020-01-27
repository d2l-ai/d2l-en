# Using Google Colab
:label:`sec_colab`

We introduced in :numref:`sec_sagemaker` and :numref:`sec_aws` for how to run this book on AWS. Another option is running on [Google Colab](https://colab.research.google.com/), which provides free GPU if you have a Google account.

To run a section on Colab, you can simply click the `Colab` button on the right of the title :numref:`fig_colab`. The first time you execute a code cell, you will receive a warning message saying it is from GitHub :numref:`fig_colab2` and may steal your data. If you trust us, then click "RUN ANYWAY", then Colab will connect you to an instance to run this notebook. In particular, if GPU is needed, such as `d2l.try_gpu()`, then we will request Colab to connect to a GPU instance automatically.

![Open a section on Colab](../img/colab.png)
:width:`400px`
:label:`fig_colab`

![The warning message for running a section on Colab](../img/colab-2.png)
:width:`400px`
:label:`fig_colab2`


## Summary

- You can use Google Colab to run each section on GPUs freely.
