```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=1}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "43027478a42d4f15867fffc50c113612",
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

# Random Search

The first method we will look at is called **random search** :cite:`bergstra-jmlr12a`. To implement random search, we pair the simple
`FIFOScheduler` with a searcher that samples configurations for new trials uniformly at random. Recall from Section
:ref:`sec_intro_search_spaces` that each hyperparameter is defined via a uniform
or loguniform distribution over its feasible range or value set. Whenever a new
configuration is to be suggested, it is drawn from these distributions.

```{.python .input  n=1}
%%tab pytorch, mxnet, tensorflow

from d2l import torch as d2l

class RandomSearcher(d2l.Searcher): #@save
    def __init__(self, search_space):
        self.save_hyperparameters()

    def sample_configuration(self):
        return {
            name: domain.sample()
            for name, domain in self.search_space.items()
        }
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

Recall the definition of our search space:

```{.python .input  n=3}
from syne_tune.search_space import loguniform, randint

search_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "batch_size": randint(8, 128)
}
```

For this search space, `RandomSearcher.sample_configuration` independently
samples `learning_rate` from `loguniform(1e-5, 1e-1)` and `batch_size` from
`randint(8, 128)`.

## The Random Search Loop

Now, we can implement the main optimization loop of random search, that iterates
until we reach the final number of iterations specified by the user. In each
iteration, we first sample a hyperparameter configuration from the subroutine
that we implemented above and then train and validate the model with the new
candidate. We also maintain the current incumbent, i.e the best configuration we
have found so far. This will be the configuration we will later return as the final
configuration. The code below makes us of `d2l.objective`, `d2l.FIFOScheduler`, and
`d2l.Tuner` as developed in the previous section, together with the specific
`RandomSearcher` we just obtained.

```{.python .input  n=4}
from d2l import torch as d2l

from syne_tune.search_space import loguniform, randint

search_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "batch_size": randint(8, 128)
}

searcher = d2l.RandomSearcher(search_space)
scheduler = d2l.FIFOScheduler(searcher=searcher)
tuner = d2l.Tuner(scheduler=scheduler, objective=d2l.objective)

tuner.run(max_wallclock_time=600)
```

Now we can plot the optimization trajectory of the incumbent to get the any-time
performance of random search:

```{.python .input  n=5}
board = d2l.ProgressBoard(xlabel='time', ylabel='error')
for time_stamp, error in zip(tuner.cumulative_runtime, tuner.incumbent_trajectory):
    board.draw(time_stamp, error, 'random search', every_n=1)
```

```{.json .output n=5}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"263.173631pt\" height=\"180.65625pt\" viewBox=\"0 0 263.173631 180.65625\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n <metadata>\n  <rdf:RDF xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2021-12-03T15:34:21.676063</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.5.0, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linejoin: round; stroke-linecap: butt}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M -0 180.65625 \nL 263.173631 180.65625 \nL 263.173631 0 \nL -0 0 \nL -0 180.65625 \nz\n\" style=\"fill: none\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 56.50625 143.1 \nL 251.80625 143.1 \nL 251.80625 7.2 \nL 56.50625 7.2 \nz\n\" style=\"fill: #ffffff\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path id=\"m9d9c712dcb\" d=\"M 0 0 \nL 0 3.5 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#m9d9c712dcb\" x=\"92.175841\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0.0001 -->\n      <g transform=\"translate(74.680529 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-30\" d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-2e\" d=\"M 684 794 \nL 1344 794 \nL 1344 0 \nL 684 0 \nL 684 794 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-31\" d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"222.65625\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"286.279297\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use xlink:href=\"#m9d9c712dcb\" x=\"140.943334\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 0.0002 -->\n      <g transform=\"translate(123.448021 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-32\" d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"222.65625\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"286.279297\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use xlink:href=\"#m9d9c712dcb\" x=\"189.710826\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 0.0003 -->\n      <g transform=\"translate(172.215514 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-33\" d=\"M 2597 2516 \nQ 3050 2419 3304 2112 \nQ 3559 1806 3559 1356 \nQ 3559 666 3084 287 \nQ 2609 -91 1734 -91 \nQ 1441 -91 1130 -33 \nQ 819 25 488 141 \nL 488 750 \nQ 750 597 1062 519 \nQ 1375 441 1716 441 \nQ 2309 441 2620 675 \nQ 2931 909 2931 1356 \nQ 2931 1769 2642 2001 \nQ 2353 2234 1838 2234 \nL 1294 2234 \nL 1294 2753 \nL 1863 2753 \nQ 2328 2753 2575 2939 \nQ 2822 3125 2822 3475 \nQ 2822 3834 2567 4026 \nQ 2313 4219 1838 4219 \nQ 1578 4219 1281 4162 \nQ 984 4106 628 3988 \nL 628 4550 \nQ 988 4650 1302 4700 \nQ 1616 4750 1894 4750 \nQ 2613 4750 3031 4423 \nQ 3450 4097 3450 3541 \nQ 3450 3153 3228 2886 \nQ 3006 2619 2597 2516 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"222.65625\"/>\n       <use xlink:href=\"#DejaVuSans-33\" x=\"286.279297\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use xlink:href=\"#m9d9c712dcb\" x=\"238.478319\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 0.0004 -->\n      <g transform=\"translate(220.983006 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-34\" d=\"M 2419 4116 \nL 825 1625 \nL 2419 1625 \nL 2419 4116 \nz\nM 2253 4666 \nL 3047 4666 \nL 3047 1625 \nL 3713 1625 \nL 3713 1100 \nL 3047 1100 \nL 3047 0 \nL 2419 0 \nL 2419 1100 \nL 313 1100 \nL 313 1709 \nL 2253 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"222.65625\"/>\n       <use xlink:href=\"#DejaVuSans-34\" x=\"286.279297\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_5\">\n     <!-- time -->\n     <g transform=\"translate(142.860156 171.376563)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-74\" d=\"M 1172 4494 \nL 1172 3500 \nL 2356 3500 \nL 2356 3053 \nL 1172 3053 \nL 1172 1153 \nQ 1172 725 1289 603 \nQ 1406 481 1766 481 \nL 2356 481 \nL 2356 0 \nL 1766 0 \nQ 1100 0 847 248 \nQ 594 497 594 1153 \nL 594 3053 \nL 172 3053 \nL 172 3500 \nL 594 3500 \nL 594 4494 \nL 1172 4494 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-69\" d=\"M 603 3500 \nL 1178 3500 \nL 1178 0 \nL 603 0 \nL 603 3500 \nz\nM 603 4863 \nL 1178 4863 \nL 1178 4134 \nL 603 4134 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6d\" d=\"M 3328 2828 \nQ 3544 3216 3844 3400 \nQ 4144 3584 4550 3584 \nQ 5097 3584 5394 3201 \nQ 5691 2819 5691 2113 \nL 5691 0 \nL 5113 0 \nL 5113 2094 \nQ 5113 2597 4934 2840 \nQ 4756 3084 4391 3084 \nQ 3944 3084 3684 2787 \nQ 3425 2491 3425 1978 \nL 3425 0 \nL 2847 0 \nL 2847 2094 \nQ 2847 2600 2669 2842 \nQ 2491 3084 2119 3084 \nQ 1678 3084 1418 2786 \nQ 1159 2488 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1356 3278 1631 3431 \nQ 1906 3584 2284 3584 \nQ 2666 3584 2933 3390 \nQ 3200 3197 3328 2828 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-65\" d=\"M 3597 1894 \nL 3597 1613 \nL 953 1613 \nQ 991 1019 1311 708 \nQ 1631 397 2203 397 \nQ 2534 397 2845 478 \nQ 3156 559 3463 722 \nL 3463 178 \nQ 3153 47 2828 -22 \nQ 2503 -91 2169 -91 \nQ 1331 -91 842 396 \nQ 353 884 353 1716 \nQ 353 2575 817 3079 \nQ 1281 3584 2069 3584 \nQ 2775 3584 3186 3129 \nQ 3597 2675 3597 1894 \nz\nM 3022 2063 \nQ 3016 2534 2758 2815 \nQ 2500 3097 2075 3097 \nQ 1594 3097 1305 2825 \nQ 1016 2553 972 2059 \nL 3022 2063 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"39.208984\"/>\n      <use xlink:href=\"#DejaVuSans-6d\" x=\"66.992188\"/>\n      <use xlink:href=\"#DejaVuSans-65\" x=\"164.404297\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_5\">\n      <defs>\n       <path id=\"mba0df7fb15\" d=\"M 0 0 \nL -3.5 0 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#mba0df7fb15\" x=\"56.50625\" y=\"124.568182\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 0.096 -->\n      <g transform=\"translate(20.878125 128.367401)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-39\" d=\"M 703 97 \nL 703 672 \nQ 941 559 1184 500 \nQ 1428 441 1663 441 \nQ 2288 441 2617 861 \nQ 2947 1281 2994 2138 \nQ 2813 1869 2534 1725 \nQ 2256 1581 1919 1581 \nQ 1219 1581 811 2004 \nQ 403 2428 403 3163 \nQ 403 3881 828 4315 \nQ 1253 4750 1959 4750 \nQ 2769 4750 3195 4129 \nQ 3622 3509 3622 2328 \nQ 3622 1225 3098 567 \nQ 2575 -91 1691 -91 \nQ 1453 -91 1209 -44 \nQ 966 3 703 97 \nz\nM 1959 2075 \nQ 2384 2075 2632 2365 \nQ 2881 2656 2881 3163 \nQ 2881 3666 2632 3958 \nQ 2384 4250 1959 4250 \nQ 1534 4250 1286 3958 \nQ 1038 3666 1038 3163 \nQ 1038 2656 1286 2365 \nQ 1534 2075 1959 2075 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-36\" d=\"M 2113 2584 \nQ 1688 2584 1439 2293 \nQ 1191 2003 1191 1497 \nQ 1191 994 1439 701 \nQ 1688 409 2113 409 \nQ 2538 409 2786 701 \nQ 3034 994 3034 1497 \nQ 3034 2003 2786 2293 \nQ 2538 2584 2113 2584 \nz\nM 3366 4563 \nL 3366 3988 \nQ 3128 4100 2886 4159 \nQ 2644 4219 2406 4219 \nQ 1781 4219 1451 3797 \nQ 1122 3375 1075 2522 \nQ 1259 2794 1537 2939 \nQ 1816 3084 2150 3084 \nQ 2853 3084 3261 2657 \nQ 3669 2231 3669 1497 \nQ 3669 778 3244 343 \nQ 2819 -91 2113 -91 \nQ 1303 -91 875 529 \nQ 447 1150 447 2328 \nQ 447 3434 972 4092 \nQ 1497 4750 2381 4750 \nQ 2619 4750 2861 4703 \nQ 3103 4656 3366 4563 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-39\" x=\"159.033203\"/>\n       <use xlink:href=\"#DejaVuSans-36\" x=\"222.65625\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_6\">\n      <g>\n       <use xlink:href=\"#mba0df7fb15\" x=\"56.50625\" y=\"99.859091\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.098 -->\n      <g transform=\"translate(20.878125 103.65831)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-38\" d=\"M 2034 2216 \nQ 1584 2216 1326 1975 \nQ 1069 1734 1069 1313 \nQ 1069 891 1326 650 \nQ 1584 409 2034 409 \nQ 2484 409 2743 651 \nQ 3003 894 3003 1313 \nQ 3003 1734 2745 1975 \nQ 2488 2216 2034 2216 \nz\nM 1403 2484 \nQ 997 2584 770 2862 \nQ 544 3141 544 3541 \nQ 544 4100 942 4425 \nQ 1341 4750 2034 4750 \nQ 2731 4750 3128 4425 \nQ 3525 4100 3525 3541 \nQ 3525 3141 3298 2862 \nQ 3072 2584 2669 2484 \nQ 3125 2378 3379 2068 \nQ 3634 1759 3634 1313 \nQ 3634 634 3220 271 \nQ 2806 -91 2034 -91 \nQ 1263 -91 848 271 \nQ 434 634 434 1313 \nQ 434 1759 690 2068 \nQ 947 2378 1403 2484 \nz\nM 1172 3481 \nQ 1172 3119 1398 2916 \nQ 1625 2713 2034 2713 \nQ 2441 2713 2670 2916 \nQ 2900 3119 2900 3481 \nQ 2900 3844 2670 4047 \nQ 2441 4250 2034 4250 \nQ 1625 4250 1398 4047 \nQ 1172 3844 1172 3481 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-39\" x=\"159.033203\"/>\n       <use xlink:href=\"#DejaVuSans-38\" x=\"222.65625\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_7\">\n      <g>\n       <use xlink:href=\"#mba0df7fb15\" x=\"56.50625\" y=\"75.15\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.100 -->\n      <g transform=\"translate(20.878125 78.949219)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"222.65625\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_8\">\n      <g>\n       <use xlink:href=\"#mba0df7fb15\" x=\"56.50625\" y=\"50.440909\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.102 -->\n      <g transform=\"translate(20.878125 54.240128)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n       <use xlink:href=\"#DejaVuSans-32\" x=\"222.65625\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_9\">\n      <g>\n       <use xlink:href=\"#mba0df7fb15\" x=\"56.50625\" y=\"25.731818\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 0.104 -->\n      <g transform=\"translate(20.878125 29.531037)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-31\" x=\"95.410156\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"159.033203\"/>\n       <use xlink:href=\"#DejaVuSans-34\" x=\"222.65625\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_11\">\n     <!-- error -->\n     <g transform=\"translate(14.798438 87.252344)rotate(-90)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-72\" d=\"M 2631 2963 \nQ 2534 3019 2420 3045 \nQ 2306 3072 2169 3072 \nQ 1681 3072 1420 2755 \nQ 1159 2438 1159 1844 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1341 3275 1631 3429 \nQ 1922 3584 2338 3584 \nQ 2397 3584 2469 3576 \nQ 2541 3569 2628 3553 \nL 2631 2963 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6f\" d=\"M 1959 3097 \nQ 1497 3097 1228 2736 \nQ 959 2375 959 1747 \nQ 959 1119 1226 758 \nQ 1494 397 1959 397 \nQ 2419 397 2687 759 \nQ 2956 1122 2956 1747 \nQ 2956 2369 2687 2733 \nQ 2419 3097 1959 3097 \nz\nM 1959 3584 \nQ 2709 3584 3137 3096 \nQ 3566 2609 3566 1747 \nQ 3566 888 3137 398 \nQ 2709 -91 1959 -91 \nQ 1206 -91 779 398 \nQ 353 888 353 1747 \nQ 353 2609 779 3096 \nQ 1206 3584 1959 3584 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-65\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"61.523438\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"100.886719\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"139.75\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"200.931641\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_10\">\n    <path d=\"M 164.329942 75.15 \n\" clip-path=\"url(#pca188366a3)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_11\">\n    <path d=\"M 164.329942 75.15 \nL 92.125798 75.15 \n\" clip-path=\"url(#pca188366a3)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_12\">\n    <path d=\"M 164.329942 75.15 \nL 92.125798 75.15 \nL 86.893614 75.15 \n\" clip-path=\"url(#pca188366a3)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_13\">\n    <path d=\"M 164.329942 75.15 \nL 92.125798 75.15 \nL 86.893614 75.15 \nL 77.010599 75.15 \n\" clip-path=\"url(#pca188366a3)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_14\">\n    <path d=\"M 164.329942 75.15 \nL 92.125798 75.15 \nL 86.893614 75.15 \nL 77.010599 75.15 \nL 72.708581 75.15 \n\" clip-path=\"url(#pca188366a3)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_15\">\n    <path d=\"M 164.329942 75.15 \nL 92.125798 75.15 \nL 86.893614 75.15 \nL 77.010599 75.15 \nL 72.708581 75.15 \nL 66.42996 75.15 \n\" clip-path=\"url(#pca188366a3)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_16\">\n    <path d=\"M 164.329942 75.15 \nL 92.125798 75.15 \nL 86.893614 75.15 \nL 77.010599 75.15 \nL 72.708581 75.15 \nL 66.42996 75.15 \nL 65.383523 75.15 \n\" clip-path=\"url(#pca188366a3)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_17\">\n    <path d=\"M 164.329942 75.15 \nL 92.125798 75.15 \nL 86.893614 75.15 \nL 77.010599 75.15 \nL 72.708581 75.15 \nL 66.42996 75.15 \nL 65.383523 75.15 \nL 81.428888 75.15 \n\" clip-path=\"url(#pca188366a3)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_18\">\n    <path d=\"M 164.329942 75.15 \nL 92.125798 75.15 \nL 86.893614 75.15 \nL 77.010599 75.15 \nL 72.708581 75.15 \nL 66.42996 75.15 \nL 65.383523 75.15 \nL 81.428888 75.15 \nL 242.928977 75.15 \n\" clip-path=\"url(#pca188366a3)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_19\">\n    <path d=\"M 164.329942 75.15 \nL 92.125798 75.15 \nL 86.893614 75.15 \nL 77.010599 75.15 \nL 72.708581 75.15 \nL 66.42996 75.15 \nL 65.383523 75.15 \nL 81.428888 75.15 \nL 242.928977 75.15 \nL 80.033639 75.15 \n\" clip-path=\"url(#pca188366a3)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 56.50625 143.1 \nL 56.50625 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 251.80625 143.1 \nL 251.80625 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 56.50625 143.1 \nL 251.80625 143.1 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 56.50625 7.2 \nL 251.80625 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 137.63125 29.878125 \nL 244.80625 29.878125 \nQ 246.80625 29.878125 246.80625 27.878125 \nL 246.80625 14.2 \nQ 246.80625 12.2 244.80625 12.2 \nL 137.63125 12.2 \nQ 135.63125 12.2 135.63125 14.2 \nL 135.63125 27.878125 \nQ 135.63125 29.878125 137.63125 29.878125 \nz\n\" style=\"fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter\"/>\n    </g>\n    <g id=\"line2d_20\">\n     <path d=\"M 139.63125 20.298438 \nL 149.63125 20.298438 \nL 159.63125 20.298438 \n\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n    </g>\n    <g id=\"text_12\">\n     <!-- random search -->\n     <g transform=\"translate(167.63125 23.798438)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-61\" d=\"M 2194 1759 \nQ 1497 1759 1228 1600 \nQ 959 1441 959 1056 \nQ 959 750 1161 570 \nQ 1363 391 1709 391 \nQ 2188 391 2477 730 \nQ 2766 1069 2766 1631 \nL 2766 1759 \nL 2194 1759 \nz\nM 3341 1997 \nL 3341 0 \nL 2766 0 \nL 2766 531 \nQ 2569 213 2275 61 \nQ 1981 -91 1556 -91 \nQ 1019 -91 701 211 \nQ 384 513 384 1019 \nQ 384 1609 779 1909 \nQ 1175 2209 1959 2209 \nL 2766 2209 \nL 2766 2266 \nQ 2766 2663 2505 2880 \nQ 2244 3097 1772 3097 \nQ 1472 3097 1187 3025 \nQ 903 2953 641 2809 \nL 641 3341 \nQ 956 3463 1253 3523 \nQ 1550 3584 1831 3584 \nQ 2591 3584 2966 3190 \nQ 3341 2797 3341 1997 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6e\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-64\" d=\"M 2906 2969 \nL 2906 4863 \nL 3481 4863 \nL 3481 0 \nL 2906 0 \nL 2906 525 \nQ 2725 213 2448 61 \nQ 2172 -91 1784 -91 \nQ 1150 -91 751 415 \nQ 353 922 353 1747 \nQ 353 2572 751 3078 \nQ 1150 3584 1784 3584 \nQ 2172 3584 2448 3432 \nQ 2725 3281 2906 2969 \nz\nM 947 1747 \nQ 947 1113 1208 752 \nQ 1469 391 1925 391 \nQ 2381 391 2643 752 \nQ 2906 1113 2906 1747 \nQ 2906 2381 2643 2742 \nQ 2381 3103 1925 3103 \nQ 1469 3103 1208 2742 \nQ 947 2381 947 1747 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-20\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-73\" d=\"M 2834 3397 \nL 2834 2853 \nQ 2591 2978 2328 3040 \nQ 2066 3103 1784 3103 \nQ 1356 3103 1142 2972 \nQ 928 2841 928 2578 \nQ 928 2378 1081 2264 \nQ 1234 2150 1697 2047 \nL 1894 2003 \nQ 2506 1872 2764 1633 \nQ 3022 1394 3022 966 \nQ 3022 478 2636 193 \nQ 2250 -91 1575 -91 \nQ 1294 -91 989 -36 \nQ 684 19 347 128 \nL 347 722 \nQ 666 556 975 473 \nQ 1284 391 1588 391 \nQ 1994 391 2212 530 \nQ 2431 669 2431 922 \nQ 2431 1156 2273 1281 \nQ 2116 1406 1581 1522 \nL 1381 1569 \nQ 847 1681 609 1914 \nQ 372 2147 372 2553 \nQ 372 3047 722 3315 \nQ 1072 3584 1716 3584 \nQ 2034 3584 2315 3537 \nQ 2597 3491 2834 3397 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-63\" d=\"M 3122 3366 \nL 3122 2828 \nQ 2878 2963 2633 3030 \nQ 2388 3097 2138 3097 \nQ 1578 3097 1268 2742 \nQ 959 2388 959 1747 \nQ 959 1106 1268 751 \nQ 1578 397 2138 397 \nQ 2388 397 2633 464 \nQ 2878 531 3122 666 \nL 3122 134 \nQ 2881 22 2623 -34 \nQ 2366 -91 2075 -91 \nQ 1284 -91 818 406 \nQ 353 903 353 1747 \nQ 353 2603 823 3093 \nQ 1294 3584 2113 3584 \nQ 2378 3584 2631 3529 \nQ 2884 3475 3122 3366 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-68\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 4863 \nL 1159 4863 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-72\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"41.113281\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"102.392578\"/>\n      <use xlink:href=\"#DejaVuSans-64\" x=\"165.771484\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"229.248047\"/>\n      <use xlink:href=\"#DejaVuSans-6d\" x=\"290.429688\"/>\n      <use xlink:href=\"#DejaVuSans-20\" x=\"387.841797\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"419.628906\"/>\n      <use xlink:href=\"#DejaVuSans-65\" x=\"471.728516\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"533.251953\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"594.53125\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"633.394531\"/>\n      <use xlink:href=\"#DejaVuSans-68\" x=\"688.375\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"pca188366a3\">\n   <rect x=\"56.50625\" y=\"7.2\" width=\"195.3\" height=\"135.9\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

![Anytime performance random search](img/anytime_performance_rs.png)
:width:`400px`
:label:`anytime_performance_rs`


## Summary

## Excercise