```{.python .input  n=2}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=2}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "9faddeb5140545a59f32bd5ee63a6176",
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

# An API for Hyperparameter Optimization
:label:`sec_api_hpo`


Before we dive into different techniques, we will first discuss a basic code structure that allows us to efficiently implement various HPO algorithms. In general, all HPO
algorithms considered here need to implement two decision making primitives, **searching** and **scheduling**. First, they need to sample new hyperparameter configurations, which often involves some kind of search over the configuration space. Additional, for each configuration an HPO algorithm needs to schedule its evaluation and how much resources should be allocated to it.
Once we started to evaluate a configurations, we will refer to it as a **trial**. We map these decisions to two classes,
`HPOSearcher` and `HPOScheduler`. On top of that, we also provide a `HPOTuner` class that executes the optimization process.

This concept of scheduler and searcher is also implemented in popular HPO libraries, such as Syne Tune, Ray Tune or Optuna.

### Searcher

Below we define a base class for searchers, which provides a new candidate
configuration through the `sample_configuration` function. A trivial way to to implement this function, would be to sample configuration uniformly at random, as we did for Random Search in the previous Section :ref:`sec_what_is_hpo`. More sophisticated algorithms, such as Bayesian optimization :ref:`sec_bo` will make these decisions based on the performance of previous trials. As a result, these more sophisticated algorithms do not sample uniformly at random, but are able to sample more promising candidates over time. To update the history of previous trials, such that we can exploit these observation to adjust our sample distribution, we add the `update` function.

```{.python .input  n=3}
%%tab mxnet
from d2l import mxnet as d2l
```

```{.json .output n=3}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"None\" cell."
 }
]
```

```{.python .input  n=4}
%%tab pytorch
from d2l import torch as d2l
```

```{.json .output n=4}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"None\" cell."
 }
]
```

```{.python .input  n=5}
%%tab tensorflow
from d2l import tensorflow as d2l
```

```{.json .output n=5}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"None\" cell."
 }
]
```

```{.python .input  n=2}
%%tab all

class HPOSearcher(d2l.HyperParameters): #@save
    def sample_configuration():
        raise NotImplementedError
    def update(self, config, error, additional_info=None):
        pass
```

The following code shows how to implement our random search optimizer from the previous section in this API:

```{.python .input  n=3}
%%tab all

class RandomSearcher(d2l.HPOSearcher): #@save
    def __init__(self, search_space):
        self.save_hyperparameters()

    def sample_configuration(self):
        return {
            name: domain.rvs()
            for name, domain in self.search_space.items()
        }
```

```{.json .output n=3}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"None\" cell."
 }
]
```

### Scheduler

Beyond sampling configurations for new trials, we also need to decide how long to
run a trial for, or whether to stop it early. In practice, all these decisions are
done by the `HPOScheduler`, who delegates the choice of new configurations to a
`HPOSearcher`. The `suggest` method is called whenever some resource for training
becomes available. Apart from invoking `sample_configuration` of a searcher, it
may also decide upon parameters like `max_epochs` (i.e., how long to train the
model for). The `update` method is called whenever a trial returns a new
observation.

```{.python .input  n=4}
%%tab all

class HPOScheduler(d2l.HyperParameters): #@save
    def suggest(self):
        raise NotImplementedError
    
    def update(self, config, error, info=None):
        raise NotImplementedError
```

To implement Random Search, but also other HPO algorithms, we only need a basic scheduler that schedules a new configuration every time new resources become available.

```{.python .input  n=5}
%%tab all

class BasicScheduler(HPOScheduler): #@save
    def __init__(self, searcher):
        self.save_hyperparameters()
        
    def suggest(self):
        return self.searcher.sample_configuration()

    def update(self, config, error, info=None):
        searcher.update(config, error, additional_info=info)
```

### Tuner

Finally, we need a component that runs the scheduler / searcher and does some book-keeping
of the results. The following code implements a sequential execution of the HPO trials that evaluates one training job after the next and will serve as a basic example. We will later use **Syne Tune** for more sophisticated distributed HPO cases.

```{.python .input  n=6}
%%tab pytorch

class HPOTuner(d2l.HyperParameters): #@save
    def __init__(self, scheduler, objective):
        self.save_hyperparameters()
        
        # bookeeping results for plotting
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.cumulative_runtime = []
        self.current_runtime = 0
        
    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
        
            error = self.objective(config)
        
            self.scheduler.update(config, error)
        
            runtime = time.time() - start_time
            self.bookkeeping(config, error.cpu().numpy(), runtime)        
```

### Bookkeeping the Performance of HPO Algorithms

With any HPO algorithm, we are mostly interested in the best performing
configuration (called **incumbent**) and its validation error after a given 
wall-clock time. This is why we track `runtime` per iteration, which includes
both the time to run an evaluation (call of `objective`) and the time to
make a decision (call of `scheduler.suggest`). In the sequel, we will plot
`cumulative_runtime` against `incumbent_trajectory` in  order to visualize the
**any-time performance** of the HPO algorithm defined in  terms of `scheduler`
(and `searcher`). This allows us to quantify not only how well the configuration
found by an optimizer works, but also how quickly an optimizer is able to find it.

```{.python .input  n=7}
%%tab pytorch

@d2l.add_to_class(HPOTuner) #@save
def bookkeeping(self, config, error, runtime): 
    # check if the last hyperparameter configuration performs better 
    # than the incumbent
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error
        
    # add current best observed performance to the optimization trajectory
    self.incumbent_trajectory.append(self.incumbent_error)
    
    # update runtime
    self.current_runtime += runtime
    self.cumulative_runtime.append(self.current_runtime)
```

### Example: Optimizing the Hyperparameters of a Convolutional Neural Network


We now use our new implementation of random search to optimize the *batch size* and *learning rate* of a convolutional neural networks from :ref:`sec_alexnet`. For that, we first have to define the objective function.

```{.python .input  n=1}
%%tab all

def objective(batch_size, learning_rate, max_epochs=8): #@save
    model = d2l.AlexNet(lr=learning_rate)
    trainer = d2l.Trainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(224, 224))
    trainer.fit(model=model, data=data)
    validation_error = trainer.validate(model=model)
    return validation_error    
```

```{.json .output n=1}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "UsageError: Cell magic `%%tab` not found.\n"
 }
]
```

We also define need to define the search space.

```{.python .input  n=15}
from scipy import stats

search_space = {
   "learning_rate": stats.loguniform(1e-4, 1),
   "batch_size": stats.randint(8, 128),
} 
```

Now we can start our random search:

```{.python .input}
searcher = d2l.RandomSearcher(search_space)
scheduler = BasicScheduler(searcher=searcher)
tuner = d2l.HPOTuner(scheduler=scheduler, objective=objective)

tuner.run(number_of_trials=5)
```

Below we plot the optimization trajectory of the incumbent to get the any-time
performance of random search:

```{.python .input  n=11}
board = d2l.ProgressBoard(xlabel='time', ylabel='error')
for time_stamp, error in zip(tuner.cumulative_runtime, tuner.incumbent_trajectory):
    board.draw(time_stamp, error, 'random search', every_n=1)
```

```{.json .output n=11}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"252.64375pt\" height=\"180.65625pt\" viewBox=\"0 0 252.64375 180.65625\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n <metadata>\n  <rdf:RDF xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2022-07-04T15:16:00.957485</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.5.0, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linejoin: round; stroke-linecap: butt}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 252.64375 180.65625 \nL 252.64375 0 \nL 0 0 \nL 0 180.65625 \nz\n\" style=\"fill: none\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 50.14375 143.1 \nL 245.44375 143.1 \nL 245.44375 7.2 \nL 50.14375 7.2 \nz\n\" style=\"fill: #ffffff\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path id=\"m1039cdf35b\" d=\"M 0 0 \nL 0 3.5 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#m1039cdf35b\" x=\"57.849359\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 100 -->\n      <g transform=\"translate(48.305609 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-31\" d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-30\" d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"127.246094\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use xlink:href=\"#m1039cdf35b\" x=\"98.174581\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 200 -->\n      <g transform=\"translate(88.630831 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-32\" d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-32\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"127.246094\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use xlink:href=\"#m1039cdf35b\" x=\"138.499803\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 300 -->\n      <g transform=\"translate(128.956053 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-33\" d=\"M 2597 2516 \nQ 3050 2419 3304 2112 \nQ 3559 1806 3559 1356 \nQ 3559 666 3084 287 \nQ 2609 -91 1734 -91 \nQ 1441 -91 1130 -33 \nQ 819 25 488 141 \nL 488 750 \nQ 750 597 1062 519 \nQ 1375 441 1716 441 \nQ 2309 441 2620 675 \nQ 2931 909 2931 1356 \nQ 2931 1769 2642 2001 \nQ 2353 2234 1838 2234 \nL 1294 2234 \nL 1294 2753 \nL 1863 2753 \nQ 2328 2753 2575 2939 \nQ 2822 3125 2822 3475 \nQ 2822 3834 2567 4026 \nQ 2313 4219 1838 4219 \nQ 1578 4219 1281 4162 \nQ 984 4106 628 3988 \nL 628 4550 \nQ 988 4650 1302 4700 \nQ 1616 4750 1894 4750 \nQ 2613 4750 3031 4423 \nQ 3450 4097 3450 3541 \nQ 3450 3153 3228 2886 \nQ 3006 2619 2597 2516 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-33\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"127.246094\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use xlink:href=\"#m1039cdf35b\" x=\"178.825025\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 400 -->\n      <g transform=\"translate(169.281275 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-34\" d=\"M 2419 4116 \nL 825 1625 \nL 2419 1625 \nL 2419 4116 \nz\nM 2253 4666 \nL 3047 4666 \nL 3047 1625 \nL 3713 1625 \nL 3713 1100 \nL 3047 1100 \nL 3047 0 \nL 2419 0 \nL 2419 1100 \nL 313 1100 \nL 313 1709 \nL 2253 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-34\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"127.246094\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use xlink:href=\"#m1039cdf35b\" x=\"219.150248\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 500 -->\n      <g transform=\"translate(209.606498 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-35\" d=\"M 691 4666 \nL 3169 4666 \nL 3169 4134 \nL 1269 4134 \nL 1269 2991 \nQ 1406 3038 1543 3061 \nQ 1681 3084 1819 3084 \nQ 2600 3084 3056 2656 \nQ 3513 2228 3513 1497 \nQ 3513 744 3044 326 \nQ 2575 -91 1722 -91 \nQ 1428 -91 1123 -41 \nQ 819 9 494 109 \nL 494 744 \nQ 775 591 1075 516 \nQ 1375 441 1709 441 \nQ 2250 441 2565 725 \nQ 2881 1009 2881 1497 \nQ 2881 1984 2565 2268 \nQ 2250 2553 1709 2553 \nQ 1456 2553 1204 2497 \nQ 953 2441 691 2322 \nL 691 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-35\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"127.246094\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_6\">\n     <!-- time -->\n     <g transform=\"translate(136.497656 171.376563)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-74\" d=\"M 1172 4494 \nL 1172 3500 \nL 2356 3500 \nL 2356 3053 \nL 1172 3053 \nL 1172 1153 \nQ 1172 725 1289 603 \nQ 1406 481 1766 481 \nL 2356 481 \nL 2356 0 \nL 1766 0 \nQ 1100 0 847 248 \nQ 594 497 594 1153 \nL 594 3053 \nL 172 3053 \nL 172 3500 \nL 594 3500 \nL 594 4494 \nL 1172 4494 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-69\" d=\"M 603 3500 \nL 1178 3500 \nL 1178 0 \nL 603 0 \nL 603 3500 \nz\nM 603 4863 \nL 1178 4863 \nL 1178 4134 \nL 603 4134 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6d\" d=\"M 3328 2828 \nQ 3544 3216 3844 3400 \nQ 4144 3584 4550 3584 \nQ 5097 3584 5394 3201 \nQ 5691 2819 5691 2113 \nL 5691 0 \nL 5113 0 \nL 5113 2094 \nQ 5113 2597 4934 2840 \nQ 4756 3084 4391 3084 \nQ 3944 3084 3684 2787 \nQ 3425 2491 3425 1978 \nL 3425 0 \nL 2847 0 \nL 2847 2094 \nQ 2847 2600 2669 2842 \nQ 2491 3084 2119 3084 \nQ 1678 3084 1418 2786 \nQ 1159 2488 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1356 3278 1631 3431 \nQ 1906 3584 2284 3584 \nQ 2666 3584 2933 3390 \nQ 3200 3197 3328 2828 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-65\" d=\"M 3597 1894 \nL 3597 1613 \nL 953 1613 \nQ 991 1019 1311 708 \nQ 1631 397 2203 397 \nQ 2534 397 2845 478 \nQ 3156 559 3463 722 \nL 3463 178 \nQ 3153 47 2828 -22 \nQ 2503 -91 2169 -91 \nQ 1331 -91 842 396 \nQ 353 884 353 1716 \nQ 353 2575 817 3079 \nQ 1281 3584 2069 3584 \nQ 2775 3584 3186 3129 \nQ 3597 2675 3597 1894 \nz\nM 3022 2063 \nQ 3016 2534 2758 2815 \nQ 2500 3097 2075 3097 \nQ 1594 3097 1305 2825 \nQ 1016 2553 972 2059 \nL 3022 2063 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"39.208984\"/>\n      <use xlink:href=\"#DejaVuSans-6d\" x=\"66.992188\"/>\n      <use xlink:href=\"#DejaVuSans-65\" x=\"164.404297\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_6\">\n      <defs>\n       <path id=\"ma9322663e5\" d=\"M 0 0 \nL -3.5 0 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#ma9322663e5\" x=\"50.14375\" y=\"122.495973\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.17 -->\n      <g transform=\"translate(20.878125 126.295192)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-2e\" d=\"M 684 794 \nL 1344 794 \nL 1344 0 \nL 684 0 \nL 684 794 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-37\" d=\"M 525 4666 \nL 3525 4666 \nL 3525 4397 \nL 1831 0 \nL 1172 0 \nL 2766 4134 \nL 525 4134 \nL 525 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-37\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_7\">\n      <g>\n       <use xlink:href=\"#ma9322663e5\" x=\"50.14375\" y=\"96.26552\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.18 -->\n      <g transform=\"translate(20.878125 100.064738)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-38\" d=\"M 2034 2216 \nQ 1584 2216 1326 1975 \nQ 1069 1734 1069 1313 \nQ 1069 891 1326 650 \nQ 1584 409 2034 409 \nQ 2484 409 2743 651 \nQ 3003 894 3003 1313 \nQ 3003 1734 2745 1975 \nQ 2488 2216 2034 2216 \nz\nM 1403 2484 \nQ 997 2584 770 2862 \nQ 544 3141 544 3541 \nQ 544 4100 942 4425 \nQ 1341 4750 2034 4750 \nQ 2731 4750 3128 4425 \nQ 3525 4100 3525 3541 \nQ 3525 3141 3298 2862 \nQ 3072 2584 2669 2484 \nQ 3125 2378 3379 2068 \nQ 3634 1759 3634 1313 \nQ 3634 634 3220 271 \nQ 2806 -91 2034 -91 \nQ 1263 -91 848 271 \nQ 434 634 434 1313 \nQ 434 1759 690 2068 \nQ 947 2378 1403 2484 \nz\nM 1172 3481 \nQ 1172 3119 1398 2916 \nQ 1625 2713 2034 2713 \nQ 2441 2713 2670 2916 \nQ 2900 3119 2900 3481 \nQ 2900 3844 2670 4047 \nQ 2441 4250 2034 4250 \nQ 1625 4250 1398 4047 \nQ 1172 3844 1172 3481 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-38\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_8\">\n      <g>\n       <use xlink:href=\"#ma9322663e5\" x=\"50.14375\" y=\"70.035066\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.19 -->\n      <g transform=\"translate(20.878125 73.834285)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-39\" d=\"M 703 97 \nL 703 672 \nQ 941 559 1184 500 \nQ 1428 441 1663 441 \nQ 2288 441 2617 861 \nQ 2947 1281 2994 2138 \nQ 2813 1869 2534 1725 \nQ 2256 1581 1919 1581 \nQ 1219 1581 811 2004 \nQ 403 2428 403 3163 \nQ 403 3881 828 4315 \nQ 1253 4750 1959 4750 \nQ 2769 4750 3195 4129 \nQ 3622 3509 3622 2328 \nQ 3622 1225 3098 567 \nQ 2575 -91 1691 -91 \nQ 1453 -91 1209 -44 \nQ 966 3 703 97 \nz\nM 1959 2075 \nQ 2384 2075 2632 2365 \nQ 2881 2656 2881 3163 \nQ 2881 3666 2632 3958 \nQ 2384 4250 1959 4250 \nQ 1534 4250 1286 3958 \nQ 1038 3666 1038 3163 \nQ 1038 2656 1286 2365 \nQ 1534 2075 1959 2075 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-39\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_9\">\n      <g>\n       <use xlink:href=\"#ma9322663e5\" x=\"50.14375\" y=\"43.804613\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 0.20 -->\n      <g transform=\"translate(20.878125 47.603832)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_10\">\n      <g>\n       <use xlink:href=\"#ma9322663e5\" x=\"50.14375\" y=\"17.57416\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 0.21 -->\n      <g transform=\"translate(20.878125 21.373379)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_12\">\n     <!-- error -->\n     <g transform=\"translate(14.798437 87.252344)rotate(-90)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-72\" d=\"M 2631 2963 \nQ 2534 3019 2420 3045 \nQ 2306 3072 2169 3072 \nQ 1681 3072 1420 2755 \nQ 1159 2438 1159 1844 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1341 3275 1631 3429 \nQ 1922 3584 2338 3584 \nQ 2397 3584 2469 3576 \nQ 2541 3569 2628 3553 \nL 2631 2963 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6f\" d=\"M 1959 3097 \nQ 1497 3097 1228 2736 \nQ 959 2375 959 1747 \nQ 959 1119 1226 758 \nQ 1494 397 1959 397 \nQ 2419 397 2687 759 \nQ 2956 1122 2956 1747 \nQ 2956 2369 2687 2733 \nQ 2419 3097 1959 3097 \nz\nM 1959 3584 \nQ 2709 3584 3137 3096 \nQ 3566 2609 3566 1747 \nQ 3566 888 3137 398 \nQ 2709 -91 1959 -91 \nQ 1206 -91 779 398 \nQ 353 888 353 1747 \nQ 353 2609 779 3096 \nQ 1206 3584 1959 3584 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-65\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"61.523438\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"100.886719\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"139.75\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"200.931641\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_11\">\n    <path d=\"M 59.021023 13.377273 \n\" clip-path=\"url(#p5cd2fd3480)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_12\">\n    <path d=\"M 59.021023 13.377273 \nL 102.403023 136.922727 \n\" clip-path=\"url(#p5cd2fd3480)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_13\">\n    <path d=\"M 59.021023 13.377273 \nL 102.403023 136.922727 \nL 146.511423 136.922727 \n\" clip-path=\"url(#p5cd2fd3480)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_14\">\n    <path d=\"M 59.021023 13.377273 \nL 102.403023 136.922727 \nL 146.511423 136.922727 \nL 191.233866 136.922727 \n\" clip-path=\"url(#p5cd2fd3480)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_15\">\n    <path d=\"M 59.021023 13.377273 \nL 102.403023 136.922727 \nL 146.511423 136.922727 \nL 191.233866 136.922727 \nL 236.566477 136.922727 \n\" clip-path=\"url(#p5cd2fd3480)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 50.14375 143.1 \nL 50.14375 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 245.44375 143.1 \nL 245.44375 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 50.14375 143.1 \nL 245.44375 143.1 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 50.14375 7.2 \nL 245.44375 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 131.26875 29.878125 \nL 238.44375 29.878125 \nQ 240.44375 29.878125 240.44375 27.878125 \nL 240.44375 14.2 \nQ 240.44375 12.2 238.44375 12.2 \nL 131.26875 12.2 \nQ 129.26875 12.2 129.26875 14.2 \nL 129.26875 27.878125 \nQ 129.26875 29.878125 131.26875 29.878125 \nz\n\" style=\"fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter\"/>\n    </g>\n    <g id=\"line2d_16\">\n     <path d=\"M 133.26875 20.298437 \nL 143.26875 20.298437 \nL 153.26875 20.298437 \n\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n    </g>\n    <g id=\"text_13\">\n     <!-- random search -->\n     <g transform=\"translate(161.26875 23.798437)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-61\" d=\"M 2194 1759 \nQ 1497 1759 1228 1600 \nQ 959 1441 959 1056 \nQ 959 750 1161 570 \nQ 1363 391 1709 391 \nQ 2188 391 2477 730 \nQ 2766 1069 2766 1631 \nL 2766 1759 \nL 2194 1759 \nz\nM 3341 1997 \nL 3341 0 \nL 2766 0 \nL 2766 531 \nQ 2569 213 2275 61 \nQ 1981 -91 1556 -91 \nQ 1019 -91 701 211 \nQ 384 513 384 1019 \nQ 384 1609 779 1909 \nQ 1175 2209 1959 2209 \nL 2766 2209 \nL 2766 2266 \nQ 2766 2663 2505 2880 \nQ 2244 3097 1772 3097 \nQ 1472 3097 1187 3025 \nQ 903 2953 641 2809 \nL 641 3341 \nQ 956 3463 1253 3523 \nQ 1550 3584 1831 3584 \nQ 2591 3584 2966 3190 \nQ 3341 2797 3341 1997 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6e\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-64\" d=\"M 2906 2969 \nL 2906 4863 \nL 3481 4863 \nL 3481 0 \nL 2906 0 \nL 2906 525 \nQ 2725 213 2448 61 \nQ 2172 -91 1784 -91 \nQ 1150 -91 751 415 \nQ 353 922 353 1747 \nQ 353 2572 751 3078 \nQ 1150 3584 1784 3584 \nQ 2172 3584 2448 3432 \nQ 2725 3281 2906 2969 \nz\nM 947 1747 \nQ 947 1113 1208 752 \nQ 1469 391 1925 391 \nQ 2381 391 2643 752 \nQ 2906 1113 2906 1747 \nQ 2906 2381 2643 2742 \nQ 2381 3103 1925 3103 \nQ 1469 3103 1208 2742 \nQ 947 2381 947 1747 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-20\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-73\" d=\"M 2834 3397 \nL 2834 2853 \nQ 2591 2978 2328 3040 \nQ 2066 3103 1784 3103 \nQ 1356 3103 1142 2972 \nQ 928 2841 928 2578 \nQ 928 2378 1081 2264 \nQ 1234 2150 1697 2047 \nL 1894 2003 \nQ 2506 1872 2764 1633 \nQ 3022 1394 3022 966 \nQ 3022 478 2636 193 \nQ 2250 -91 1575 -91 \nQ 1294 -91 989 -36 \nQ 684 19 347 128 \nL 347 722 \nQ 666 556 975 473 \nQ 1284 391 1588 391 \nQ 1994 391 2212 530 \nQ 2431 669 2431 922 \nQ 2431 1156 2273 1281 \nQ 2116 1406 1581 1522 \nL 1381 1569 \nQ 847 1681 609 1914 \nQ 372 2147 372 2553 \nQ 372 3047 722 3315 \nQ 1072 3584 1716 3584 \nQ 2034 3584 2315 3537 \nQ 2597 3491 2834 3397 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-63\" d=\"M 3122 3366 \nL 3122 2828 \nQ 2878 2963 2633 3030 \nQ 2388 3097 2138 3097 \nQ 1578 3097 1268 2742 \nQ 959 2388 959 1747 \nQ 959 1106 1268 751 \nQ 1578 397 2138 397 \nQ 2388 397 2633 464 \nQ 2878 531 3122 666 \nL 3122 134 \nQ 2881 22 2623 -34 \nQ 2366 -91 2075 -91 \nQ 1284 -91 818 406 \nQ 353 903 353 1747 \nQ 353 2603 823 3093 \nQ 1294 3584 2113 3584 \nQ 2378 3584 2631 3529 \nQ 2884 3475 3122 3366 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-68\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 4863 \nL 1159 4863 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-72\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"41.113281\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"102.392578\"/>\n      <use xlink:href=\"#DejaVuSans-64\" x=\"165.771484\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"229.248047\"/>\n      <use xlink:href=\"#DejaVuSans-6d\" x=\"290.429688\"/>\n      <use xlink:href=\"#DejaVuSans-20\" x=\"387.841797\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"419.628906\"/>\n      <use xlink:href=\"#DejaVuSans-65\" x=\"471.728516\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"533.251953\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"594.53125\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"633.394531\"/>\n      <use xlink:href=\"#DejaVuSans-68\" x=\"688.375\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p5cd2fd3480\">\n   <rect x=\"50.14375\" y=\"7.2\" width=\"195.3\" height=\"135.9\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

## Comparing HPO Algorithms

Just as with training algorithms or model architectures, it is important to understand how to best
compare different HPO algorithms. Each HPO run depends on mostly two sources of randomness:
the random effects of the training process, such as random weight initialization or mini-batch ordering, and, the intrinsic randomness of the HPO algorithm itself, such as the random sampling of random search. Hence, when comparing different algorithms, it is crucial to run each experiment several times and report statistics, such as mean or median, across a population of multiple repetitions of an algorithm based on different seeds of the random number generator.

To illustrate this, we compare random search (see :ref:'sec_rs') and Bayesian optimization, which we will discuss in Section :ref:'sec_bo', for optimizing the hyperparameters of a feed-forward neural network. Each algorithm was evaluated $50$ times with a different random seed. The solid line indicates the average performance of the incumbent across these $50$ repetitions and the dashed line the standard deviation. We can see that random search and Bayesian optimization perform roughly the same up to ~1000 seconds, but Bayesian optimization can make use of the past observation to identify better configurations and thus quickly outperforms random search afterwards.


![Example any-time performance plot to compare two algorithms A and B](img/example_anytime_performance.png)
:width:`400px`
:label:`example_anytime_performance`


## Summary

## Exercise
