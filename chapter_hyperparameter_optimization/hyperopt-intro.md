```{.python .input  n=2}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=2}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "d715e4f29e4e4bc7991b78d60657a80a",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "interactive(children=(Dropdown(description='tab', options=('mxnet', 'pytorch', 'tensorflow'), value=None), Out\u2026"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

# What is Hyperparameter Optimization?
:label:`sec_what_is_hpo`

Consider the weight parameters of a deep neural network. They are automatically determined during training by minimizing the loss function with, for example, stochastic gradient descent (see Chapter :ref:`sec_sgd`). 
But, every neural network comes with additional parameters, called **hyperparameters** that define and control this training process, for example the learning rate, batch size or determine the capacity of the network, e.g number of layers or the activation function for each layer. Unfortunately these **hyperparameters** cannot be learned in the same way as the weight parameters. For discrete or integer hyperparameters, the training error might not be differentiable with respect to them and even for continuous hyperparameters, such as the learning rate, we would need to back-propagate through the entire training process :cite:`maclaurin-icml15`.

Without a different form of automation, the user has to set them manually by trial-and-error process, in what amounts to a time-consuming and difficult part of machine learning workflows :cite:`hpo`. 
Hyperparameters are usually not directly transferable across architectures and datasets :cite:`hpo`, rendering the search for the right hyperparameters as a recurring task in practice. Furthermore, for the most hyperparameters there are no rule-of-thumbs and often expert knowledge is required to find sensible values.

Hyperparameter optimization (HPO) algorithms are designed to tackle this problem in an principled and automated fashion :cite:`hpo`. The main idea is to cast the search for the optimal hyperparameter configuration that optimizes some objective function as a global optimization problem. The objective is usually to minimize generalization error on a hold-out validation dataset, but can be in principle also the training time, inference time, model complexity or a combination of these.

More recently, hyperparameter optimization has been extended :cite:`nas` to search for entirely new neural network architectures. This is often refered to as neural architecture search (NAS) in the literature. Compared to classical HPO, NAS is often even more compute heavy and requires additional efforts to evaluate single architectures. Both HPO and NAS can be considered as sub-fields of AutoML :cite:`automl` which aims to automated the entire ML pipeline.

In this section we will introduce HPO and show how we can automatically find the right hyperparameters of an logistic regression example, introduced in :ref:`sec_softmax_concise`, that minimizes the classification error on the hold-out validation dataset. 

##  The Optimization Problem
:label:`sec_definition_hpo`

To get started we will first look at a simple toy problem: Based on the logistic classification example from :ref:`sec_sgd`, we will search for the right learning rate that leads to an high validation accuracy on the Fashion MNIST dataset. Even though there are more hyperparameters we could optimize, for example the batch size or the number of epochs, we will focus only on the learning rate to keep the complexity of this example at bay.

```{.python .input  n=3}
from d2l import torch as d2l

from torch import nn

class SoftmaxClassification(d2l.Classification):
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))
    def forward(self, X):
        return self.net(X)
```

### The Objective Function


The performance of a learning algorithm can be seen as a function $f: \mathcal{X} \rightarrow \mathbb{R}$ that maps from the hyperparameter space $\mathbf{x} \in \mathcal{X}$ to the validation error. For every evaluation of $f(\mathbf{x})$, we have to train and validate our machine learning model, which particularly for deep neural networks on large datasets, is time and compute intensive. We will see in Chapter :ref:`sec_mf_hpo` how we can exploit cheaper approximations of $f$ to speed up the optimization process. Training is stochastic in general (e.g., weights are randomly initialized, mini-batches are randomly sampled), so that our observations will be noisy: $y \sim f(\mathbf{x}) + \epsilon$, where we assume that $\epsilon \sim N(0, \sigma)$.

Now, since we would like to optimize the validation error, we need to add a method computing this metric.

```{.python .input  n=4}
%%tab all

from d2l import torch as d2l
import torch
from torch import nn

def validate(trainer, model):
    model.eval()
    accuracy = 0
    val_batch_idx = 0
    for batch in trainer.val_dataloader:
        with torch.no_grad():
            x, y = trainer.prepare_batch(batch)
            y_hat = model(x)
            accuracy += model.accuracy(y_hat, y)
        val_batch_idx += 1
    return 1 -  accuracy / val_batch_idx
```

We optimize validation error with respect to the hyperparameter configuration `config`,
consisting of the `learning_rate`. For each evaluation, we train our model
for `max_epochs` epochs, then compute and return its validation error:

```{.python .input  n=5}
# %%tab all

def hpo_objective_softmax_classification(config, max_epochs=10):
    learning_rate = config['learning_rate']
    trainer = d2l.Trainer(max_epochs=max_epochs)
    
    data = d2l.FashionMNIST(batch_size=16)
    model = SoftmaxClassification(num_outputs=10, lr=learning_rate)
    trainer.fit(model=model, data=data)
#     validation_error = trainer.validate(model=model)
    validation_error = validate(trainer, model=model)
    return validation_error.numpy()
```

Now, given our criterion $f(\mathbf{x})$ in terms of `hpo_objective_softmax_classification(config)`, where $\mathbf{x}$ corresponds to `config`, we would like to find: $\mathbf{x}_{\star} \in argmin_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$. 

Since $f$ is the validation error after training, there is no efficient way to compute gradients with respect to $\mathbf{x}$. While there is recent work :cite:`maclaurin-icml15`,`franceschi-icml17a` to drive HPO by approximate "hypergradients", none of the existing approaches are competitive with the state-of-the-art yet, and we will not discuss them here. Furthermore, the computational burden of evaluating $f$ requires a particular sample efficiency of HPO methods.

### The Search Space

:label:`sec_intro_search_spaces`

Along with the objective function $f(\mathbf{x})$, we also need to define the feasible set $\mathbf{x} \in \mathcal{X}$ to optimize over, the *search space* or *configuration space*. For our logistic regression example we define the following search space:

```{.python .input  n=6}
from syne_tune.search_space import loguniform, randint

search_space = {
   "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
} 
```

Each parameter has a data type, such as `float` for `learning_rate`, as well as a closed bounded range
(i.e., lower and upper bounds). Some positive parameters, such as `learning_rate`, are best represented on a logarithmic scale as optimal values can differ by several orders of magnitude, while others come with linear scale.

![Example search space for a simple neural network architecture](img/example_search_space.png)
:width:`40px`
:label:`example_search_spacee`


In general, the structure of the search space $\mathcal{X}$ can be complex and it can be quite different from $\mathbb{R}^d$. Beyond the scope of this chapter, some hyperparameters may
depend on the value of others. For example, if we try to tune both the number of layers and widths per layer for a multi-layer perceptron,
the width of the $l$-th layer is relevant only if the network has at least $l+1$ layers. Hence, hyperparameter optimization consists in determining a small set of good hyperparameters to search
over and probing the performance of the trained machine learning model at these values in the hope that 
one of them will be close to the best hyperparameters $\mathbf{x}_*$. 

## Grid Search

Now, we look at the first method to solve our hyperparameter optimization problem: grid search. 
The main idea of grid search is to define a discrete set of values for each hyperparameter and then to evaluate the cartesian product, i.e grid of all hyperparameters. The evaluation can be executed sequentially (as we do here) or in parallel.
Arguably because of its simplicitiy, grid search is one of the most often use methods to solve hyperparameter optimization. It doesn't require any sophisticated implementation and can be applied to any hyperparameter type.

```{.python .input  n=26}
errors = []
for lr in search_space['learning_rate']:
        errors.append(hpo_objective_softmax_classification({"learning_rate": lr}))
  
```

```{.python .input  n=13}
import numpy as np
best_idx = np.argmin(errors)
print(search_space['learning_rate'][best_idx])
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0.01\n"
 }
]
```

Below we plot the validation error of each hyperparameter configuration we just evaluated. We can see that there is a high variance in terms of validation performance across all hyperparameters.

```{.python .input  n=17}
import matplotlib.pyplot as plt

plt.figure(dpi=200)
plt.scatter(search_space['learning_rate'], errors)
plt.xscale('log')
plt.ylabel('validation error')
plt.xlabel('learning rate')
plt.show()
```

```{.json .output n=17}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"392.14375pt\" height=\"262.19625pt\" viewBox=\"0 0 392.14375 262.19625\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n <metadata>\n  <rdf:RDF xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2022-05-30T18:00:07.427219</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.5.0, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linejoin: round; stroke-linecap: butt}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 262.19625 \nL 392.14375 262.19625 \nL 392.14375 0 \nL 0 0 \nL 0 262.19625 \nz\n\" style=\"fill: none\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 50.14375 224.64 \nL 384.94375 224.64 \nL 384.94375 7.2 \nL 50.14375 7.2 \nz\n\" style=\"fill: #ffffff\"/>\n   </g>\n   <g id=\"PathCollection_1\">\n    <defs>\n     <path id=\"m1e8347f10e\" d=\"M 0 3 \nC 0.795609 3 1.55874 2.683901 2.12132 2.12132 \nC 2.683901 1.55874 3 0.795609 3 0 \nC 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 \nC 1.55874 -2.683901 0.795609 -3 0 -3 \nC -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 \nC -2.683901 -1.55874 -3 -0.795609 -3 0 \nC -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 \nC -1.55874 2.683901 -0.795609 3 0 3 \nz\n\" style=\"stroke: #1f77b4\"/>\n    </defs>\n    <g clip-path=\"url(#p108cfbbf59)\">\n     <use xlink:href=\"#m1e8347f10e\" x=\"65.361932\" y=\"17.083636\" style=\"fill: #1f77b4; stroke: #1f77b4\"/>\n     <use xlink:href=\"#m1e8347f10e\" x=\"103.407386\" y=\"174.136561\" style=\"fill: #1f77b4; stroke: #1f77b4\"/>\n     <use xlink:href=\"#m1e8347f10e\" x=\"141.452841\" y=\"214.756364\" style=\"fill: #1f77b4; stroke: #1f77b4\"/>\n     <use xlink:href=\"#m1e8347f10e\" x=\"179.498295\" y=\"131.346243\" style=\"fill: #1f77b4; stroke: #1f77b4\"/>\n     <use xlink:href=\"#m1e8347f10e\" x=\"217.54375\" y=\"157.082478\" style=\"fill: #1f77b4; stroke: #1f77b4\"/>\n     <use xlink:href=\"#m1e8347f10e\" x=\"255.589205\" y=\"155.99722\" style=\"fill: #1f77b4; stroke: #1f77b4\"/>\n     <use xlink:href=\"#m1e8347f10e\" x=\"293.634659\" y=\"174.136561\" style=\"fill: #1f77b4; stroke: #1f77b4\"/>\n     <use xlink:href=\"#m1e8347f10e\" x=\"331.680114\" y=\"192.430965\" style=\"fill: #1f77b4; stroke: #1f77b4\"/>\n     <use xlink:href=\"#m1e8347f10e\" x=\"369.725568\" y=\"142.97404\" style=\"fill: #1f77b4; stroke: #1f77b4\"/>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path id=\"m9844c0b50e\" d=\"M 0 0 \nL 0 3.5 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#m9844c0b50e\" x=\"103.407386\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- $\\mathdefault{10^{-3}}$ -->\n      <g transform=\"translate(91.657386 239.238438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-31\" d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-30\" d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-2212\" d=\"M 678 2272 \nL 4684 2272 \nL 4684 1741 \nL 678 1741 \nL 678 2272 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-33\" d=\"M 2597 2516 \nQ 3050 2419 3304 2112 \nQ 3559 1806 3559 1356 \nQ 3559 666 3084 287 \nQ 2609 -91 1734 -91 \nQ 1441 -91 1130 -33 \nQ 819 25 488 141 \nL 488 750 \nQ 750 597 1062 519 \nQ 1375 441 1716 441 \nQ 2309 441 2620 675 \nQ 2931 909 2931 1356 \nQ 2931 1769 2642 2001 \nQ 2353 2234 1838 2234 \nL 1294 2234 \nL 1294 2753 \nL 1863 2753 \nQ 2328 2753 2575 2939 \nQ 2822 3125 2822 3475 \nQ 2822 3834 2567 4026 \nQ 2313 4219 1838 4219 \nQ 1578 4219 1281 4162 \nQ 984 4106 628 3988 \nL 628 4550 \nQ 988 4650 1302 4700 \nQ 1616 4750 1894 4750 \nQ 2613 4750 3031 4423 \nQ 3450 4097 3450 3541 \nQ 3450 3153 3228 2886 \nQ 3006 2619 2597 2516 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\" transform=\"translate(0 0.765625)\"/>\n       <use xlink:href=\"#DejaVuSans-30\" transform=\"translate(63.623047 0.765625)\"/>\n       <use xlink:href=\"#DejaVuSans-2212\" transform=\"translate(128.203125 39.046875)scale(0.7)\"/>\n       <use xlink:href=\"#DejaVuSans-33\" transform=\"translate(186.855469 39.046875)scale(0.7)\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use xlink:href=\"#m9844c0b50e\" x=\"179.498295\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- $\\mathdefault{10^{-1}}$ -->\n      <g transform=\"translate(167.748295 239.238438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\" transform=\"translate(0 0.684375)\"/>\n       <use xlink:href=\"#DejaVuSans-30\" transform=\"translate(63.623047 0.684375)\"/>\n       <use xlink:href=\"#DejaVuSans-2212\" transform=\"translate(128.203125 38.965625)scale(0.7)\"/>\n       <use xlink:href=\"#DejaVuSans-31\" transform=\"translate(186.855469 38.965625)scale(0.7)\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use xlink:href=\"#m9844c0b50e\" x=\"255.589205\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- $\\mathdefault{10^{1}}$ -->\n      <g transform=\"translate(246.789205 239.238438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\" transform=\"translate(0 0.684375)\"/>\n       <use xlink:href=\"#DejaVuSans-30\" transform=\"translate(63.623047 0.684375)\"/>\n       <use xlink:href=\"#DejaVuSans-31\" transform=\"translate(128.203125 38.965625)scale(0.7)\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use xlink:href=\"#m9844c0b50e\" x=\"331.680114\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- $\\mathdefault{10^{3}}$ -->\n      <g transform=\"translate(322.880114 239.238438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\" transform=\"translate(0 0.765625)\"/>\n       <use xlink:href=\"#DejaVuSans-30\" transform=\"translate(63.623047 0.765625)\"/>\n       <use xlink:href=\"#DejaVuSans-33\" transform=\"translate(128.203125 39.046875)scale(0.7)\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_5\">\n     <!-- learning rate -->\n     <g transform=\"translate(185.4 252.916563)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-6c\" d=\"M 603 4863 \nL 1178 4863 \nL 1178 0 \nL 603 0 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-65\" d=\"M 3597 1894 \nL 3597 1613 \nL 953 1613 \nQ 991 1019 1311 708 \nQ 1631 397 2203 397 \nQ 2534 397 2845 478 \nQ 3156 559 3463 722 \nL 3463 178 \nQ 3153 47 2828 -22 \nQ 2503 -91 2169 -91 \nQ 1331 -91 842 396 \nQ 353 884 353 1716 \nQ 353 2575 817 3079 \nQ 1281 3584 2069 3584 \nQ 2775 3584 3186 3129 \nQ 3597 2675 3597 1894 \nz\nM 3022 2063 \nQ 3016 2534 2758 2815 \nQ 2500 3097 2075 3097 \nQ 1594 3097 1305 2825 \nQ 1016 2553 972 2059 \nL 3022 2063 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-61\" d=\"M 2194 1759 \nQ 1497 1759 1228 1600 \nQ 959 1441 959 1056 \nQ 959 750 1161 570 \nQ 1363 391 1709 391 \nQ 2188 391 2477 730 \nQ 2766 1069 2766 1631 \nL 2766 1759 \nL 2194 1759 \nz\nM 3341 1997 \nL 3341 0 \nL 2766 0 \nL 2766 531 \nQ 2569 213 2275 61 \nQ 1981 -91 1556 -91 \nQ 1019 -91 701 211 \nQ 384 513 384 1019 \nQ 384 1609 779 1909 \nQ 1175 2209 1959 2209 \nL 2766 2209 \nL 2766 2266 \nQ 2766 2663 2505 2880 \nQ 2244 3097 1772 3097 \nQ 1472 3097 1187 3025 \nQ 903 2953 641 2809 \nL 641 3341 \nQ 956 3463 1253 3523 \nQ 1550 3584 1831 3584 \nQ 2591 3584 2966 3190 \nQ 3341 2797 3341 1997 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-72\" d=\"M 2631 2963 \nQ 2534 3019 2420 3045 \nQ 2306 3072 2169 3072 \nQ 1681 3072 1420 2755 \nQ 1159 2438 1159 1844 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1341 3275 1631 3429 \nQ 1922 3584 2338 3584 \nQ 2397 3584 2469 3576 \nQ 2541 3569 2628 3553 \nL 2631 2963 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6e\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-69\" d=\"M 603 3500 \nL 1178 3500 \nL 1178 0 \nL 603 0 \nL 603 3500 \nz\nM 603 4863 \nL 1178 4863 \nL 1178 4134 \nL 603 4134 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-67\" d=\"M 2906 1791 \nQ 2906 2416 2648 2759 \nQ 2391 3103 1925 3103 \nQ 1463 3103 1205 2759 \nQ 947 2416 947 1791 \nQ 947 1169 1205 825 \nQ 1463 481 1925 481 \nQ 2391 481 2648 825 \nQ 2906 1169 2906 1791 \nz\nM 3481 434 \nQ 3481 -459 3084 -895 \nQ 2688 -1331 1869 -1331 \nQ 1566 -1331 1297 -1286 \nQ 1028 -1241 775 -1147 \nL 775 -588 \nQ 1028 -725 1275 -790 \nQ 1522 -856 1778 -856 \nQ 2344 -856 2625 -561 \nQ 2906 -266 2906 331 \nL 2906 616 \nQ 2728 306 2450 153 \nQ 2172 0 1784 0 \nQ 1141 0 747 490 \nQ 353 981 353 1791 \nQ 353 2603 747 3093 \nQ 1141 3584 1784 3584 \nQ 2172 3584 2450 3431 \nQ 2728 3278 2906 2969 \nL 2906 3500 \nL 3481 3500 \nL 3481 434 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-20\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-74\" d=\"M 1172 4494 \nL 1172 3500 \nL 2356 3500 \nL 2356 3053 \nL 1172 3053 \nL 1172 1153 \nQ 1172 725 1289 603 \nQ 1406 481 1766 481 \nL 2356 481 \nL 2356 0 \nL 1766 0 \nQ 1100 0 847 248 \nQ 594 497 594 1153 \nL 594 3053 \nL 172 3053 \nL 172 3500 \nL 594 3500 \nL 594 4494 \nL 1172 4494 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-6c\"/>\n      <use xlink:href=\"#DejaVuSans-65\" x=\"27.783203\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"89.306641\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"150.585938\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"189.949219\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"253.328125\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"281.111328\"/>\n      <use xlink:href=\"#DejaVuSans-67\" x=\"344.490234\"/>\n      <use xlink:href=\"#DejaVuSans-20\" x=\"407.966797\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"439.753906\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"480.867188\"/>\n      <use xlink:href=\"#DejaVuSans-74\" x=\"542.146484\"/>\n      <use xlink:href=\"#DejaVuSans-65\" x=\"581.355469\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_5\">\n      <defs>\n       <path id=\"mbc722e2722\" d=\"M 0 0 \nL -3.5 0 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#mbc722e2722\" x=\"50.14375\" y=\"222.973366\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 0.16 -->\n      <g transform=\"translate(20.878125 226.772585)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-2e\" d=\"M 684 794 \nL 1344 794 \nL 1344 0 \nL 684 0 \nL 684 794 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-36\" d=\"M 2113 2584 \nQ 1688 2584 1439 2293 \nQ 1191 2003 1191 1497 \nQ 1191 994 1439 701 \nQ 1688 409 2113 409 \nQ 2538 409 2786 701 \nQ 3034 994 3034 1497 \nQ 3034 2003 2786 2293 \nQ 2538 2584 2113 2584 \nz\nM 3366 4563 \nL 3366 3988 \nQ 3128 4100 2886 4159 \nQ 2644 4219 2406 4219 \nQ 1781 4219 1451 3797 \nQ 1122 3375 1075 2522 \nQ 1259 2794 1537 2939 \nQ 1816 3084 2150 3084 \nQ 2853 3084 3261 2657 \nQ 3669 2231 3669 1497 \nQ 3669 778 3244 343 \nQ 2819 -91 2113 -91 \nQ 1303 -91 875 529 \nQ 447 1150 447 2328 \nQ 447 3434 972 4092 \nQ 1497 4750 2381 4750 \nQ 2619 4750 2861 4703 \nQ 3103 4656 3366 4563 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-36\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_6\">\n      <g>\n       <use xlink:href=\"#mbc722e2722\" x=\"50.14375\" y=\"191.965879\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.18 -->\n      <g transform=\"translate(20.878125 195.765097)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-38\" d=\"M 2034 2216 \nQ 1584 2216 1326 1975 \nQ 1069 1734 1069 1313 \nQ 1069 891 1326 650 \nQ 1584 409 2034 409 \nQ 2484 409 2743 651 \nQ 3003 894 3003 1313 \nQ 3003 1734 2745 1975 \nQ 2488 2216 2034 2216 \nz\nM 1403 2484 \nQ 997 2584 770 2862 \nQ 544 3141 544 3541 \nQ 544 4100 942 4425 \nQ 1341 4750 2034 4750 \nQ 2731 4750 3128 4425 \nQ 3525 4100 3525 3541 \nQ 3525 3141 3298 2862 \nQ 3072 2584 2669 2484 \nQ 3125 2378 3379 2068 \nQ 3634 1759 3634 1313 \nQ 3634 634 3220 271 \nQ 2806 -91 2034 -91 \nQ 1263 -91 848 271 \nQ 434 634 434 1313 \nQ 434 1759 690 2068 \nQ 947 2378 1403 2484 \nz\nM 1172 3481 \nQ 1172 3119 1398 2916 \nQ 1625 2713 2034 2713 \nQ 2441 2713 2670 2916 \nQ 2900 3119 2900 3481 \nQ 2900 3844 2670 4047 \nQ 2441 4250 2034 4250 \nQ 1625 4250 1398 4047 \nQ 1172 3844 1172 3481 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-38\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_7\">\n      <g>\n       <use xlink:href=\"#mbc722e2722\" x=\"50.14375\" y=\"160.958391\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.20 -->\n      <g transform=\"translate(20.878125 164.75761)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-32\" d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_8\">\n      <g>\n       <use xlink:href=\"#mbc722e2722\" x=\"50.14375\" y=\"129.950904\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.22 -->\n      <g transform=\"translate(20.878125 133.750123)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_9\">\n      <g>\n       <use xlink:href=\"#mbc722e2722\" x=\"50.14375\" y=\"98.943417\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 0.24 -->\n      <g transform=\"translate(20.878125 102.742636)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-34\" d=\"M 2419 4116 \nL 825 1625 \nL 2419 1625 \nL 2419 4116 \nz\nM 2253 4666 \nL 3047 4666 \nL 3047 1625 \nL 3713 1625 \nL 3713 1100 \nL 3047 1100 \nL 3047 0 \nL 2419 0 \nL 2419 1100 \nL 313 1100 \nL 313 1709 \nL 2253 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-34\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_6\">\n     <g id=\"line2d_10\">\n      <g>\n       <use xlink:href=\"#mbc722e2722\" x=\"50.14375\" y=\"67.93593\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 0.26 -->\n      <g transform=\"translate(20.878125 71.735148)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-36\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_7\">\n     <g id=\"line2d_11\">\n      <g>\n       <use xlink:href=\"#mbc722e2722\" x=\"50.14375\" y=\"36.928443\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_12\">\n      <!-- 0.28 -->\n      <g transform=\"translate(20.878125 40.727661)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-38\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_13\">\n     <!-- validation error -->\n     <g transform=\"translate(14.798438 154.228594)rotate(-90)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-76\" d=\"M 191 3500 \nL 800 3500 \nL 1894 563 \nL 2988 3500 \nL 3597 3500 \nL 2284 0 \nL 1503 0 \nL 191 3500 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-64\" d=\"M 2906 2969 \nL 2906 4863 \nL 3481 4863 \nL 3481 0 \nL 2906 0 \nL 2906 525 \nQ 2725 213 2448 61 \nQ 2172 -91 1784 -91 \nQ 1150 -91 751 415 \nQ 353 922 353 1747 \nQ 353 2572 751 3078 \nQ 1150 3584 1784 3584 \nQ 2172 3584 2448 3432 \nQ 2725 3281 2906 2969 \nz\nM 947 1747 \nQ 947 1113 1208 752 \nQ 1469 391 1925 391 \nQ 2381 391 2643 752 \nQ 2906 1113 2906 1747 \nQ 2906 2381 2643 2742 \nQ 2381 3103 1925 3103 \nQ 1469 3103 1208 2742 \nQ 947 2381 947 1747 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6f\" d=\"M 1959 3097 \nQ 1497 3097 1228 2736 \nQ 959 2375 959 1747 \nQ 959 1119 1226 758 \nQ 1494 397 1959 397 \nQ 2419 397 2687 759 \nQ 2956 1122 2956 1747 \nQ 2956 2369 2687 2733 \nQ 2419 3097 1959 3097 \nz\nM 1959 3584 \nQ 2709 3584 3137 3096 \nQ 3566 2609 3566 1747 \nQ 3566 888 3137 398 \nQ 2709 -91 1959 -91 \nQ 1206 -91 779 398 \nQ 353 888 353 1747 \nQ 353 2609 779 3096 \nQ 1206 3584 1959 3584 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-76\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"59.179688\"/>\n      <use xlink:href=\"#DejaVuSans-6c\" x=\"120.458984\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"148.242188\"/>\n      <use xlink:href=\"#DejaVuSans-64\" x=\"176.025391\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"239.501953\"/>\n      <use xlink:href=\"#DejaVuSans-74\" x=\"300.78125\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"339.990234\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"367.773438\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"428.955078\"/>\n      <use xlink:href=\"#DejaVuSans-20\" x=\"492.333984\"/>\n      <use xlink:href=\"#DejaVuSans-65\" x=\"524.121094\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"585.644531\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"625.007812\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"663.871094\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"725.052734\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 50.14375 224.64 \nL 50.14375 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 384.94375 224.64 \nL 384.94375 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 50.14375 224.64 \nL 384.94375 224.64 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 50.14375 7.2 \nL 384.94375 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p108cfbbf59\">\n   <rect x=\"50.14375\" y=\"7.2\" width=\"334.8\" height=\"217.44\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 1200x800 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

Unfortunately grid search suffers from the curse of dimensionality. The number of configurations in our grid grows exponentially with the number of hyperparameters, rendering it often infeasible to optimize more than 3 hyperparameters jointly. 
Another caveat of grid search is that it is rather unlikely that $\mathbf{x_{\star}}$ lies exactly on one of the grid points in continuous spaces. Thus, even though grid search is often able to find a well performing configurations, it is unlikely that it will find the global optimum $\mathbf{x_{\star}}$.

In the next sections we will look at more sample efficient hyperparameter optimization methods that overcome the shortcomings of grid search and speed up the expensive optimization process substantially.

# Summary

# Exercise
