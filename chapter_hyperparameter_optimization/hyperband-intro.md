```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=1}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "7a89c4084fc84c118ca54d406bde0ddf",
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

# Multi-fidelity Hyperparameter Optimization

Most neural network models are pretty expensive to train even for a single
hyperparameter configuration. In order to converge, we need to train for
a substantial number of epochs. Without a massively parallel compute budget,
random search will often be too slow to be of real use in practice. What can
we do?

If a configuration is a poor choice, do we really have to wait for many epochs
until we know? Why not just evaluate configurations after a few epochs only,
and stop the worst ones early? In this section, we will develop some competitive
algorithms based on this **early stopping** idea.

## Early Stopping Hyperparameter Configurations

In the Figure we depict the learning curves of neural networks with different hyperparameter configuration trained for the same number
`max_epochs` of epochs. We can see that after a few epochs we are already
able to visually distinguish between the well performing and the poorly performing
configurations. However, the correlation is not perfect, and we might still require
the full amount of epochs to identify the best performing configuration.

<!-- ![Learning curves of random hyperparameter configurations](../../img/samples_lc.pdf) -->
![Learning curves of random hyperparameter configurations](img/samples_lc.png)
:width:`400px`
:label:`img_samples_lc`

Based on this observation, we can free up compute resources by early stopping
the evaluation of poorly performing configuration and allocate resources to
more promising configurations. This will eventually speed up the optimization
process, since we have a higher throughput of configurations that we can try.
More formally, we expand our definition in Section :ref:sec_definition_hpo,
such that our objective function $f(\mathbf{x}, r)$ gets an additional input
$r \in [r_{min}, r_{max}]$ that specifies the amount of resource that we are
willing to spend for the evaluation of $\mathbf{x}$. We assume that both the
correlation to $f(\mathbf{x}) = f(\mathbf{x}, r_{max})$ and the computational
cost $c(\mathbf{x}, r)$ increases with $r$ (the latter should be affine linear
in $r$). Typically, $r$ represents the number of epochs for training the neural
network. But also other resources are possible, such as the training dataset
size or the number of cross-validation folds.

```{.python .input  n=5}
from syne_tune.search_space import loguniform, uniform, randint
from d2l import torch as d2l

search_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "batch_size": randint(8, 128)
} 

def objective_with_resource(config):
    epochs = int(config['resource'])
    _config = {k:v for k, v in config.items() if k != 'resource'}
    return d2l.objective(_config, max_epochs=epochs)
```

## Successive Halving

One of the simplest ways to combine random search with early stopping is
**successive halving** :cite:`jamieson-aistats16`,`karnin-icml13`. The basic idea is to start with $N$ trials, each for a randomly chosen
configuration, and train each of them for only $r_{min}$ epochs (e.g., $r_{min} = 1$). We then
discard a fraction of the worst performing trials and train the remaining ones
for longer. Iterating this process, less and less trials run for longer and
longer, until at least one trial reaches `max_epochs` epochs.

More formally, specify a minimum budget $r_{min}$ and a halving constant
$\eta\in\{2, 3, \dots\}$. For simplicity, assume that $r_{max} = r_{min} \eta^K$,
where $r_{max}$ is equal to `max_epochs`. Moreover, set $N = \eta^K$. Let us
define *rung levels* $\mathcal{R} = \{ r_{min}, r_{min}\eta, r_{min}\eta^2,
\dots, r_{max} \}$. In general, a trial is trained until reaching a rung level,
then evaluated there, and the validation errors of all trials at a rung level
are used to decide which of them to discard. We start with running $N$ trials
until rung level $r_{min}$. Sorting the validation errors, we keep the top
$1 / \eta$ fraction ($\eta^{K-1}$ trials) and discard all the rest. The surviving
trials are trained for $r_{min}\eta$ epochs (next rung level), and the process
is repeated. In each round, a $1 / \eta$ fraction of trials
survives and finds it budget to be multiplied by $\eta$. With this particular
choice of $N$, only a single trial will be trained to the full budget of
`max_epochs` epochs. Finally, such traversals over all rung levels are
repeated in an outer loop until the total experiment budget is spent.

When implementing successive halving from scratch, we will take a shortcut.
Recall that our `objective` code allows to specify the number of epochs
to be trained as `max_epochs`, and we will make use of this, by feeding in
$r_{min}$, $r_{min}\eta$, for this argument. In particular, if a trial is chosen
to continue after $r_{min}$ or $r_{min}\eta$ epochs, its training to the next
level is started from scratch.

```{.python .input  n=3}
import numpy as np
from d2l import torch as d2l

class SuccessiveHalvingScheduler(d2l.Scheduler):
    def __init__(
            self, searcher, eta, r_min, r_max, bottom_rung_size=None):
        self.save_hyperparameters()

        if bottom_rung_size is None:
            # Default choice for successive halving
            self.bottom_rung_size = r_max
        x = r_max / r_min
        K = int(np.log(x) / np.log(eta))
        self.rung_levels = [r_min * eta ** k for k in range(K + 1)]
        
        self.rungs = {r: [] for r in self.rung_levels}
    
    def update(self, config, error, info=None):
        r = config['resource']
        self.rungs[r].append((config, error))
        self.searcher.update(config, error, additional_info=info)
    
    def get_top_n_configurations(self, rung_level, n):
        rung = self.rungs[rung_level]
        if not rung:
            return []
        configs, errors = zip(*rung)
        
        indices = [np.argsort(errors)[i] for i in range(n)]
        return [configs[i] for i in indices]

    def _rung_level_size(self, rung_index):
        return self.bottom_rung_size // (self.eta ** rung_index)

    def find_rung_level(self):
        for i, r in enumerate(self.rung_levels):
            rung_size = self._rung_level_size(i)
            if len(self.rungs[r]) < rung_size:
                return r
    
    def find_promotable_configuration(self, rung_level):
        previous_rung_level = rung_level // self.eta
        n = self._rung_level_size(self.rung_levels.index(rung_level))
        top_configs_on_previous_rung_level = self.get_top_n_configurations(
            previous_rung_level, n)
        for config in top_configs_on_previous_rung_level:
             if config not in self.rungs[rung_level]:
                return config
        
    def suggest(self):
        # If we do not have enough configurations on the smallest rung_level we sample
        # a random configuration
        n = self._rung_level_size(0)
        if len(self.rungs[self.r_min]) < n:
            config = searcher.sample_configuration()
            # Run this config until r_min only
            config['resource'] = self.r_min
            return config
        else:
            # Lowest rung level not yet complete:
            r = self.find_rung_level()
            # check which configuration we should promote to the next rung level
            config = self.find_promotable_configuration(r)        
            config['resource'] = r
            return config        
```

Let us see how this is doing on our example.

```{.python .input  n=6}
searcher = d2l.RandomSearcher(search_space)
scheduler = SuccessiveHalvingScheduler(searcher=searcher, eta=2, r_min=1, r_max=16)
tuner = d2l.Tuner(scheduler=scheduler, objective=objective_with_resource)
tuner.run(max_wallclock_time=600)
```

```{.json .output n=6}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"240.554688pt\" height=\"180.65625pt\" viewBox=\"0 0 240.554688 180.65625\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n <metadata>\n  <rdf:RDF xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2022-01-06T19:30:01.373466</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.5.0, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linejoin: round; stroke-linecap: butt}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 240.554688 180.65625 \nL 240.554688 0 \nL 0 0 \nL 0 180.65625 \nz\n\" style=\"fill: none\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \nL 225.403125 7.2 \nL 30.103125 7.2 \nz\n\" style=\"fill: #ffffff\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path id=\"m61fc0d0792\" d=\"M 0 0 \nL 0 3.5 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#m61fc0d0792\" x=\"30.103125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0.0 -->\n      <g transform=\"translate(22.151563 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-30\" d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-2e\" d=\"M 684 794 \nL 1344 794 \nL 1344 0 \nL 684 0 \nL 684 794 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use xlink:href=\"#m61fc0d0792\" x=\"69.163125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 0.2 -->\n      <g transform=\"translate(61.211563 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-32\" d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use xlink:href=\"#m61fc0d0792\" x=\"108.223125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 0.4 -->\n      <g transform=\"translate(100.271563 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-34\" d=\"M 2419 4116 \nL 825 1625 \nL 2419 1625 \nL 2419 4116 \nz\nM 2253 4666 \nL 3047 4666 \nL 3047 1625 \nL 3713 1625 \nL 3713 1100 \nL 3047 1100 \nL 3047 0 \nL 2419 0 \nL 2419 1100 \nL 313 1100 \nL 313 1709 \nL 2253 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-34\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use xlink:href=\"#m61fc0d0792\" x=\"147.283125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 0.6 -->\n      <g transform=\"translate(139.331563 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-36\" d=\"M 2113 2584 \nQ 1688 2584 1439 2293 \nQ 1191 2003 1191 1497 \nQ 1191 994 1439 701 \nQ 1688 409 2113 409 \nQ 2538 409 2786 701 \nQ 3034 994 3034 1497 \nQ 3034 2003 2786 2293 \nQ 2538 2584 2113 2584 \nz\nM 3366 4563 \nL 3366 3988 \nQ 3128 4100 2886 4159 \nQ 2644 4219 2406 4219 \nQ 1781 4219 1451 3797 \nQ 1122 3375 1075 2522 \nQ 1259 2794 1537 2939 \nQ 1816 3084 2150 3084 \nQ 2853 3084 3261 2657 \nQ 3669 2231 3669 1497 \nQ 3669 778 3244 343 \nQ 2819 -91 2113 -91 \nQ 1303 -91 875 529 \nQ 447 1150 447 2328 \nQ 447 3434 972 4092 \nQ 1497 4750 2381 4750 \nQ 2619 4750 2861 4703 \nQ 3103 4656 3366 4563 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-36\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use xlink:href=\"#m61fc0d0792\" x=\"186.343125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 0.8 -->\n      <g transform=\"translate(178.391563 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-38\" d=\"M 2034 2216 \nQ 1584 2216 1326 1975 \nQ 1069 1734 1069 1313 \nQ 1069 891 1326 650 \nQ 1584 409 2034 409 \nQ 2484 409 2743 651 \nQ 3003 894 3003 1313 \nQ 3003 1734 2745 1975 \nQ 2488 2216 2034 2216 \nz\nM 1403 2484 \nQ 997 2584 770 2862 \nQ 544 3141 544 3541 \nQ 544 4100 942 4425 \nQ 1341 4750 2034 4750 \nQ 2731 4750 3128 4425 \nQ 3525 4100 3525 3541 \nQ 3525 3141 3298 2862 \nQ 3072 2584 2669 2484 \nQ 3125 2378 3379 2068 \nQ 3634 1759 3634 1313 \nQ 3634 634 3220 271 \nQ 2806 -91 2034 -91 \nQ 1263 -91 848 271 \nQ 434 634 434 1313 \nQ 434 1759 690 2068 \nQ 947 2378 1403 2484 \nz\nM 1172 3481 \nQ 1172 3119 1398 2916 \nQ 1625 2713 2034 2713 \nQ 2441 2713 2670 2916 \nQ 2900 3119 2900 3481 \nQ 2900 3844 2670 4047 \nQ 2441 4250 2034 4250 \nQ 1625 4250 1398 4047 \nQ 1172 3844 1172 3481 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-38\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_6\">\n      <g>\n       <use xlink:href=\"#m61fc0d0792\" x=\"225.403125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 1.0 -->\n      <g transform=\"translate(217.451563 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-31\" d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_7\">\n     <!-- epoch -->\n     <g transform=\"translate(112.525 171.376563)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-65\" d=\"M 3597 1894 \nL 3597 1613 \nL 953 1613 \nQ 991 1019 1311 708 \nQ 1631 397 2203 397 \nQ 2534 397 2845 478 \nQ 3156 559 3463 722 \nL 3463 178 \nQ 3153 47 2828 -22 \nQ 2503 -91 2169 -91 \nQ 1331 -91 842 396 \nQ 353 884 353 1716 \nQ 353 2575 817 3079 \nQ 1281 3584 2069 3584 \nQ 2775 3584 3186 3129 \nQ 3597 2675 3597 1894 \nz\nM 3022 2063 \nQ 3016 2534 2758 2815 \nQ 2500 3097 2075 3097 \nQ 1594 3097 1305 2825 \nQ 1016 2553 972 2059 \nL 3022 2063 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-70\" d=\"M 1159 525 \nL 1159 -1331 \nL 581 -1331 \nL 581 3500 \nL 1159 3500 \nL 1159 2969 \nQ 1341 3281 1617 3432 \nQ 1894 3584 2278 3584 \nQ 2916 3584 3314 3078 \nQ 3713 2572 3713 1747 \nQ 3713 922 3314 415 \nQ 2916 -91 2278 -91 \nQ 1894 -91 1617 61 \nQ 1341 213 1159 525 \nz\nM 3116 1747 \nQ 3116 2381 2855 2742 \nQ 2594 3103 2138 3103 \nQ 1681 3103 1420 2742 \nQ 1159 2381 1159 1747 \nQ 1159 1113 1420 752 \nQ 1681 391 2138 391 \nQ 2594 391 2855 752 \nQ 3116 1113 3116 1747 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6f\" d=\"M 1959 3097 \nQ 1497 3097 1228 2736 \nQ 959 2375 959 1747 \nQ 959 1119 1226 758 \nQ 1494 397 1959 397 \nQ 2419 397 2687 759 \nQ 2956 1122 2956 1747 \nQ 2956 2369 2687 2733 \nQ 2419 3097 1959 3097 \nz\nM 1959 3584 \nQ 2709 3584 3137 3096 \nQ 3566 2609 3566 1747 \nQ 3566 888 3137 398 \nQ 2709 -91 1959 -91 \nQ 1206 -91 779 398 \nQ 353 888 353 1747 \nQ 353 2609 779 3096 \nQ 1206 3584 1959 3584 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-63\" d=\"M 3122 3366 \nL 3122 2828 \nQ 2878 2963 2633 3030 \nQ 2388 3097 2138 3097 \nQ 1578 3097 1268 2742 \nQ 959 2388 959 1747 \nQ 959 1106 1268 751 \nQ 1578 397 2138 397 \nQ 2388 397 2633 464 \nQ 2878 531 3122 666 \nL 3122 134 \nQ 2881 22 2623 -34 \nQ 2366 -91 2075 -91 \nQ 1284 -91 818 406 \nQ 353 903 353 1747 \nQ 353 2603 823 3093 \nQ 1294 3584 2113 3584 \nQ 2378 3584 2631 3529 \nQ 2884 3475 3122 3366 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-68\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 4863 \nL 1159 4863 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-65\"/>\n      <use xlink:href=\"#DejaVuSans-70\" x=\"61.523438\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"125\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"186.181641\"/>\n      <use xlink:href=\"#DejaVuSans-68\" x=\"241.162109\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_7\">\n      <defs>\n       <path id=\"mfafcf1ad4b\" d=\"M 0 0 \nL -3.5 0 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#mfafcf1ad4b\" x=\"30.103125\" y=\"142.524746\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.0 -->\n      <g transform=\"translate(7.2 146.323965)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_8\">\n      <g>\n       <use xlink:href=\"#mfafcf1ad4b\" x=\"30.103125\" y=\"114.481107\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.5 -->\n      <g transform=\"translate(7.2 118.280326)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-35\" d=\"M 691 4666 \nL 3169 4666 \nL 3169 4134 \nL 1269 4134 \nL 1269 2991 \nQ 1406 3038 1543 3061 \nQ 1681 3084 1819 3084 \nQ 2600 3084 3056 2656 \nQ 3513 2228 3513 1497 \nQ 3513 744 3044 326 \nQ 2575 -91 1722 -91 \nQ 1428 -91 1123 -41 \nQ 819 9 494 109 \nL 494 744 \nQ 775 591 1075 516 \nQ 1375 441 1709 441 \nQ 2250 441 2565 725 \nQ 2881 1009 2881 1497 \nQ 2881 1984 2565 2268 \nQ 2250 2553 1709 2553 \nQ 1456 2553 1204 2497 \nQ 953 2441 691 2322 \nL 691 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-35\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_9\">\n      <g>\n       <use xlink:href=\"#mfafcf1ad4b\" x=\"30.103125\" y=\"86.437468\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 1.0 -->\n      <g transform=\"translate(7.2 90.236687)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_10\">\n      <g>\n       <use xlink:href=\"#mfafcf1ad4b\" x=\"30.103125\" y=\"58.393829\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 1.5 -->\n      <g transform=\"translate(7.2 62.193048)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-35\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_11\">\n      <g>\n       <use xlink:href=\"#mfafcf1ad4b\" x=\"30.103125\" y=\"30.350191\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_12\">\n      <!-- 2.0 -->\n      <g transform=\"translate(7.2 34.149409)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-32\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_12\">\n    <path d=\"M 78.850005 13.377273 \n\" clip-path=\"url(#pb94e1967c0)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_13\">\n    <path d=\"M 78.850005 13.377273 \nL 176.500005 13.397626 \n\" clip-path=\"url(#pb94e1967c0)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_14\">\n    <path d=\"M 78.850005 13.377273 \nL 176.500005 13.397626 \n\" clip-path=\"url(#pb94e1967c0)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_15\">\n    <path d=\"M 225.403125 13.401239 \n\" clip-path=\"url(#pb94e1967c0)\" style=\"fill: none; stroke-dasharray: 5.55,2.4; stroke-dashoffset: 0; stroke: #ff7f0e; stroke-width: 1.5\"/>\n   </g>\n   <g id=\"line2d_16\"/>\n   <g id=\"line2d_17\">\n    <path d=\"M 78.850005 13.377273 \nL 176.500005 13.397626 \n\" clip-path=\"url(#pb94e1967c0)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_18\">\n    <path d=\"M 225.403125 13.401239 \n\" clip-path=\"url(#pb94e1967c0)\" style=\"fill: none; stroke-dasharray: 5.55,2.4; stroke-dashoffset: 0; stroke: #ff7f0e; stroke-width: 1.5\"/>\n   </g>\n   <g id=\"line2d_19\">\n    <path d=\"M 225.403125 136.922727 \n\" clip-path=\"url(#pb94e1967c0)\" style=\"fill: none; stroke-dasharray: 9.6,2.4,1.5,2.4; stroke-dashoffset: 0; stroke: #2ca02c; stroke-width: 1.5\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 30.103125 143.1 \nL 30.103125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 30.103125 7.2 \nL 225.403125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 37.103125 138.1 \nL 116.69375 138.1 \nQ 118.69375 138.1 118.69375 136.1 \nL 118.69375 92.23125 \nQ 118.69375 90.23125 116.69375 90.23125 \nL 37.103125 90.23125 \nQ 35.103125 90.23125 35.103125 92.23125 \nL 35.103125 136.1 \nQ 35.103125 138.1 37.103125 138.1 \nz\n\" style=\"fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter\"/>\n    </g>\n    <g id=\"line2d_20\">\n     <path d=\"M 39.103125 98.329687 \nL 49.103125 98.329687 \nL 59.103125 98.329687 \n\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n    </g>\n    <g id=\"text_13\">\n     <!-- train_loss -->\n     <g transform=\"translate(67.103125 101.829687)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-74\" d=\"M 1172 4494 \nL 1172 3500 \nL 2356 3500 \nL 2356 3053 \nL 1172 3053 \nL 1172 1153 \nQ 1172 725 1289 603 \nQ 1406 481 1766 481 \nL 2356 481 \nL 2356 0 \nL 1766 0 \nQ 1100 0 847 248 \nQ 594 497 594 1153 \nL 594 3053 \nL 172 3053 \nL 172 3500 \nL 594 3500 \nL 594 4494 \nL 1172 4494 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-72\" d=\"M 2631 2963 \nQ 2534 3019 2420 3045 \nQ 2306 3072 2169 3072 \nQ 1681 3072 1420 2755 \nQ 1159 2438 1159 1844 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1341 3275 1631 3429 \nQ 1922 3584 2338 3584 \nQ 2397 3584 2469 3576 \nQ 2541 3569 2628 3553 \nL 2631 2963 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-61\" d=\"M 2194 1759 \nQ 1497 1759 1228 1600 \nQ 959 1441 959 1056 \nQ 959 750 1161 570 \nQ 1363 391 1709 391 \nQ 2188 391 2477 730 \nQ 2766 1069 2766 1631 \nL 2766 1759 \nL 2194 1759 \nz\nM 3341 1997 \nL 3341 0 \nL 2766 0 \nL 2766 531 \nQ 2569 213 2275 61 \nQ 1981 -91 1556 -91 \nQ 1019 -91 701 211 \nQ 384 513 384 1019 \nQ 384 1609 779 1909 \nQ 1175 2209 1959 2209 \nL 2766 2209 \nL 2766 2266 \nQ 2766 2663 2505 2880 \nQ 2244 3097 1772 3097 \nQ 1472 3097 1187 3025 \nQ 903 2953 641 2809 \nL 641 3341 \nQ 956 3463 1253 3523 \nQ 1550 3584 1831 3584 \nQ 2591 3584 2966 3190 \nQ 3341 2797 3341 1997 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-69\" d=\"M 603 3500 \nL 1178 3500 \nL 1178 0 \nL 603 0 \nL 603 3500 \nz\nM 603 4863 \nL 1178 4863 \nL 1178 4134 \nL 603 4134 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6e\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-5f\" d=\"M 3263 -1063 \nL 3263 -1509 \nL -63 -1509 \nL -63 -1063 \nL 3263 -1063 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6c\" d=\"M 603 4863 \nL 1178 4863 \nL 1178 0 \nL 603 0 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-73\" d=\"M 2834 3397 \nL 2834 2853 \nQ 2591 2978 2328 3040 \nQ 2066 3103 1784 3103 \nQ 1356 3103 1142 2972 \nQ 928 2841 928 2578 \nQ 928 2378 1081 2264 \nQ 1234 2150 1697 2047 \nL 1894 2003 \nQ 2506 1872 2764 1633 \nQ 3022 1394 3022 966 \nQ 3022 478 2636 193 \nQ 2250 -91 1575 -91 \nQ 1294 -91 989 -36 \nQ 684 19 347 128 \nL 347 722 \nQ 666 556 975 473 \nQ 1284 391 1588 391 \nQ 1994 391 2212 530 \nQ 2431 669 2431 922 \nQ 2431 1156 2273 1281 \nQ 2116 1406 1581 1522 \nL 1381 1569 \nQ 847 1681 609 1914 \nQ 372 2147 372 2553 \nQ 372 3047 722 3315 \nQ 1072 3584 1716 3584 \nQ 2034 3584 2315 3537 \nQ 2597 3491 2834 3397 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"39.208984\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"80.322266\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"141.601562\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"169.384766\"/>\n      <use xlink:href=\"#DejaVuSans-5f\" x=\"232.763672\"/>\n      <use xlink:href=\"#DejaVuSans-6c\" x=\"282.763672\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"310.546875\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"371.728516\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"423.828125\"/>\n     </g>\n    </g>\n    <g id=\"line2d_21\">\n     <path d=\"M 39.103125 113.285937 \nL 49.103125 113.285937 \nL 59.103125 113.285937 \n\" style=\"fill: none; stroke-dasharray: 5.55,2.4; stroke-dashoffset: 0; stroke: #ff7f0e; stroke-width: 1.5\"/>\n    </g>\n    <g id=\"text_14\">\n     <!-- val_loss -->\n     <g transform=\"translate(67.103125 116.785937)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-76\" d=\"M 191 3500 \nL 800 3500 \nL 1894 563 \nL 2988 3500 \nL 3597 3500 \nL 2284 0 \nL 1503 0 \nL 191 3500 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-76\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"59.179688\"/>\n      <use xlink:href=\"#DejaVuSans-6c\" x=\"120.458984\"/>\n      <use xlink:href=\"#DejaVuSans-5f\" x=\"148.242188\"/>\n      <use xlink:href=\"#DejaVuSans-6c\" x=\"198.242188\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"226.025391\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"287.207031\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"339.306641\"/>\n     </g>\n    </g>\n    <g id=\"line2d_22\">\n     <path d=\"M 39.103125 128.242188 \nL 49.103125 128.242188 \nL 59.103125 128.242188 \n\" style=\"fill: none; stroke-dasharray: 9.6,2.4,1.5,2.4; stroke-dashoffset: 0; stroke: #2ca02c; stroke-width: 1.5\"/>\n    </g>\n    <g id=\"text_15\">\n     <!-- val_acc -->\n     <g transform=\"translate(67.103125 131.742188)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-76\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"59.179688\"/>\n      <use xlink:href=\"#DejaVuSans-6c\" x=\"120.458984\"/>\n      <use xlink:href=\"#DejaVuSans-5f\" x=\"148.242188\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"198.242188\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"259.521484\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"314.501953\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"pb94e1967c0\">\n   <rect x=\"30.103125\" y=\"7.2\" width=\"195.3\" height=\"135.9\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

```{.python .input  n=7}
import matplotlib.pyplot as plt
for r in scheduler.rungs:
    rung = scheduler.rungs[r]
    for i in rung:    
        plt.scatter(r, i[1])
    plt.xlim(0, 17)
        
```

```{.json .output n=7}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"378.465625pt\" height=\"248.518125pt\" viewBox=\"0 0 378.465625 248.518125\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n <metadata>\n  <rdf:RDF xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2022-01-06T19:30:01.495818</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.5.0, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linejoin: round; stroke-linecap: butt}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 248.518125 \nL 378.465625 248.518125 \nL 378.465625 0 \nL 0 0 \nL 0 248.518125 \nz\n\" style=\"fill: none\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 36.465625 224.64 \nL 371.265625 224.64 \nL 371.265625 7.2 \nL 36.465625 7.2 \nz\n\" style=\"fill: #ffffff\"/>\n   </g>\n   <g id=\"PathCollection_1\">\n    <defs>\n     <path id=\"m16b02a948c\" d=\"M 0 3 \nC 0.795609 3 1.55874 2.683901 2.12132 2.12132 \nC 2.683901 1.55874 3 0.795609 3 0 \nC 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 \nC 1.55874 -2.683901 0.795609 -3 0 -3 \nC -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 \nC -2.683901 -1.55874 -3 -0.795609 -3 0 \nC -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 \nC -1.55874 2.683901 -0.795609 3 0 3 \nz\n\" style=\"stroke: #1f77b4\"/>\n    </defs>\n    <g clip-path=\"url(#pc2758e94fb)\">\n     <use xlink:href=\"#m16b02a948c\" x=\"56.159743\" y=\"115.92\" style=\"fill: #1f77b4; stroke: #1f77b4\"/>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path id=\"m25410a30b5\" d=\"M 0 0 \nL 0 3.5 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#m25410a30b5\" x=\"36.465625\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0 -->\n      <g transform=\"translate(33.284375 239.238437)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-30\" d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use xlink:href=\"#m25410a30b5\" x=\"75.85386\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 2 -->\n      <g transform=\"translate(72.67261 239.238437)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-32\" d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-32\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use xlink:href=\"#m25410a30b5\" x=\"115.242096\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 4 -->\n      <g transform=\"translate(112.060846 239.238437)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-34\" d=\"M 2419 4116 \nL 825 1625 \nL 2419 1625 \nL 2419 4116 \nz\nM 2253 4666 \nL 3047 4666 \nL 3047 1625 \nL 3713 1625 \nL 3713 1100 \nL 3047 1100 \nL 3047 0 \nL 2419 0 \nL 2419 1100 \nL 313 1100 \nL 313 1709 \nL 2253 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-34\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use xlink:href=\"#m25410a30b5\" x=\"154.630331\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 6 -->\n      <g transform=\"translate(151.449081 239.238437)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-36\" d=\"M 2113 2584 \nQ 1688 2584 1439 2293 \nQ 1191 2003 1191 1497 \nQ 1191 994 1439 701 \nQ 1688 409 2113 409 \nQ 2538 409 2786 701 \nQ 3034 994 3034 1497 \nQ 3034 2003 2786 2293 \nQ 2538 2584 2113 2584 \nz\nM 3366 4563 \nL 3366 3988 \nQ 3128 4100 2886 4159 \nQ 2644 4219 2406 4219 \nQ 1781 4219 1451 3797 \nQ 1122 3375 1075 2522 \nQ 1259 2794 1537 2939 \nQ 1816 3084 2150 3084 \nQ 2853 3084 3261 2657 \nQ 3669 2231 3669 1497 \nQ 3669 778 3244 343 \nQ 2819 -91 2113 -91 \nQ 1303 -91 875 529 \nQ 447 1150 447 2328 \nQ 447 3434 972 4092 \nQ 1497 4750 2381 4750 \nQ 2619 4750 2861 4703 \nQ 3103 4656 3366 4563 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-36\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use xlink:href=\"#m25410a30b5\" x=\"194.018566\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 8 -->\n      <g transform=\"translate(190.837316 239.238437)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-38\" d=\"M 2034 2216 \nQ 1584 2216 1326 1975 \nQ 1069 1734 1069 1313 \nQ 1069 891 1326 650 \nQ 1584 409 2034 409 \nQ 2484 409 2743 651 \nQ 3003 894 3003 1313 \nQ 3003 1734 2745 1975 \nQ 2488 2216 2034 2216 \nz\nM 1403 2484 \nQ 997 2584 770 2862 \nQ 544 3141 544 3541 \nQ 544 4100 942 4425 \nQ 1341 4750 2034 4750 \nQ 2731 4750 3128 4425 \nQ 3525 4100 3525 3541 \nQ 3525 3141 3298 2862 \nQ 3072 2584 2669 2484 \nQ 3125 2378 3379 2068 \nQ 3634 1759 3634 1313 \nQ 3634 634 3220 271 \nQ 2806 -91 2034 -91 \nQ 1263 -91 848 271 \nQ 434 634 434 1313 \nQ 434 1759 690 2068 \nQ 947 2378 1403 2484 \nz\nM 1172 3481 \nQ 1172 3119 1398 2916 \nQ 1625 2713 2034 2713 \nQ 2441 2713 2670 2916 \nQ 2900 3119 2900 3481 \nQ 2900 3844 2670 4047 \nQ 2441 4250 2034 4250 \nQ 1625 4250 1398 4047 \nQ 1172 3844 1172 3481 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-38\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_6\">\n      <g>\n       <use xlink:href=\"#m25410a30b5\" x=\"233.406801\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 10 -->\n      <g transform=\"translate(227.044301 239.238437)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-31\" d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_7\">\n     <g id=\"line2d_7\">\n      <g>\n       <use xlink:href=\"#m25410a30b5\" x=\"272.795037\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 12 -->\n      <g transform=\"translate(266.432537 239.238437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"63.623047\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_8\">\n     <g id=\"line2d_8\">\n      <g>\n       <use xlink:href=\"#m25410a30b5\" x=\"312.183272\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 14 -->\n      <g transform=\"translate(305.820772 239.238437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-34\" x=\"63.623047\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_9\">\n     <g id=\"line2d_9\">\n      <g>\n       <use xlink:href=\"#m25410a30b5\" x=\"351.571507\" y=\"224.64\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 16 -->\n      <g transform=\"translate(345.209007 239.238437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-36\" x=\"63.623047\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_10\">\n      <defs>\n       <path id=\"mc94e4b1b87\" d=\"M 0 0 \nL -3.5 0 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#mc94e4b1b87\" x=\"36.465625\" y=\"203.663163\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 1.10 -->\n      <g transform=\"translate(7.2 207.462382)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-2e\" d=\"M 684 794 \nL 1344 794 \nL 1344 0 \nL 684 0 \nL 684 794 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_11\">\n      <g>\n       <use xlink:href=\"#mc94e4b1b87\" x=\"36.465625\" y=\"169.317998\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 1.12 -->\n      <g transform=\"translate(7.2 173.117216)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_12\">\n      <g>\n       <use xlink:href=\"#mc94e4b1b87\" x=\"36.465625\" y=\"134.972832\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_12\">\n      <!-- 1.14 -->\n      <g transform=\"translate(7.2 138.772051)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-34\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_13\">\n      <g>\n       <use xlink:href=\"#mc94e4b1b87\" x=\"36.465625\" y=\"100.627666\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_13\">\n      <!-- 1.16 -->\n      <g transform=\"translate(7.2 104.426885)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-36\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_14\">\n      <g>\n       <use xlink:href=\"#mc94e4b1b87\" x=\"36.465625\" y=\"66.282501\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_14\">\n      <!-- 1.18 -->\n      <g transform=\"translate(7.2 70.08172)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-38\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_6\">\n     <g id=\"line2d_15\">\n      <g>\n       <use xlink:href=\"#mc94e4b1b87\" x=\"36.465625\" y=\"31.937335\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_15\">\n      <!-- 1.20 -->\n      <g transform=\"translate(7.2 35.736554)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 36.465625 224.64 \nL 36.465625 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 371.265625 224.64 \nL 371.265625 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 36.465625 224.64 \nL 371.265625 224.64 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 36.465625 7.2 \nL 371.265625 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"pc2758e94fb\">\n   <rect x=\"36.465625\" y=\"7.2\" width=\"334.8\" height=\"217.44\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

## Hyperband

While successive halving can greatly improve upon random search, the choice of $r_{min}$
can be critical for good performance. If $r_{min}$ is too small, validation errors at the
lowest rung may be determined more by initial random weights than by the hyperparameters,
and even the best configurations may be filtered out at random. If $r_{min}$ is too large on the other hand,
the benefits of early stopping may be greatly diminished.

Hyperband :cite:`li-iclr17` is an extension of successive halving (SH) which mitigates the risk of setting
$r_{min}$ too small. Recall that SH defines $K + 1$ rung levels $\mathcal{R} =
\{ r_{min}, r_{min}\eta, \dots, r_{max} \}$. Hyperband runs SH as a subroutine, based on
brackets $\mathcal{R}_b = \{ r_{min}\eta^b, r_{min}\eta^{b+1},\dots, r_{max}\}$, $b=0,
\dots, K$. Each bracket comes with an initial number $N_b$ of trials, which are chosen at
random and trained for $r_{min}\eta^b$ epochs. The $N_b$ are selected such that the total
resources (i.e., number of epochs trained) in each bracket are about the same. Now,
Hyperband runs SH on the brackets in a round robin fashion. On bracket $b=0$, we have SH
with the original choice $r_{min}$, while on bracket $b=K$, we run standard random
search, since the only rung level is $r_{max}$.

![Learning curves based on Hyperband](img/hyperband.jpeg)
:width:`400px`
:label:`img_samples_hb`

Here is code for Hyperband scheduling.

*MS: There is something wrong in this code. The number $N_b$ of trials in the
lowest rung needs to be set depending on the bracket, and that is not done
here. I think `SuccessiveHalvingScheduler` needs some argument for that, which
could be None and set by default, but for Hyperband, you need to pass it.*

*MS: I implemented what I think is a fix, but please do check this!*

```{.python .input  n=8}
import numpy as np
import copy

class HyperbandScheduler(d2l.Scheduler):
    def __init__(self, searcher, eta, r_min, r_max):
        self.save_hyperparameters()
        self.current_bracket = 0
        self.brackets = []
        self.s_max = int(np.ceil(
            (np.log(r_max) - np.log(r_min)) / np.log(eta)))
        self.successive_halving = self._create_successive_halving()

    def _create_successive_halving(self):
        # Determine size of bottom rung
        r_num_m1 = self.s_max - self.current_bracket
        pre_fact = (self.s_max + 1) / (r_num_m1 + 1)
        bottom_rung_size = int(np.ceil(
            pre_fact * self.eta ** r_num_m1))
        r_min = self.r_min * self.eta ** self.current_bracket
        return SuccessiveHalvingScheduler(
            searcher=self.searcher, eta=self.eta, r_min=r_min,
            r_max=self.r_max, bottom_rung_size=bottom_rung_size)
  
    def suggest(self):
        # check if we finished the current bracket
        r_max = self.successive_halving.r_max
        r_min = self.successive_halving.r_min
        if len(self.successive_halving.rungs[r_max]) == r_min:
            self.current_bracket += 1
            
            r_min = self.r_min * self.eta ** self.current_bracket
            
            if r_min > self.r_max:
                r_min = self.r_min
                self.current_bracket = 0
            print(r_min)
            # start a new bracket with a higher r_min
            self.brackets.append(copy.deepcopy(self.successive_halving.rungs))
            self.successive_halving = self._create_successive_halving()

        return self.successive_halving.suggest()
        
    def update(self, config, error, info=None):
        self.successive_halving.update(config, error, info=info)
```

Let us see how this is doing.

```{.python .input  n=9}
searcher = d2l.RandomSearcher(search_space)
scheduler = HyperbandScheduler(searcher=searcher, eta=2, r_min=1, r_max=16)
tuner = d2l.Tuner(scheduler=scheduler, objective=objective_with_resource)
tuner.run(max_wallclock_time=600)
```

```{.python .input  n=82}
import matplotlib.pyplot as plt

for bracket in scheduler.brackets:
    for r in bracket:
        rung = bracket[r]
        for i in rung:       
            
            plt.scatter(r, i[1])
        plt.xlim(0, 17)
    plt.show()
```

## Summary


## Exercises
