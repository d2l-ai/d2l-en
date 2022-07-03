# `d2l` Document API
:label:`sec_d2l` 

Les implémentations des membres suivants du paquetage `d2l` et les sections où ils sont définis et expliqués peuvent être trouvés dans le [source file](https://github.com/d2l-ai/d2l-en/tree/master/d2l).


:begin_tab:`mxnet`

```eval_rst

.. currentmodule:: d2l.mxnet

```


:end_tab:

:begin_tab:`pytorch`

```eval_rst

.. currentmodule:: d2l.torch

```


:begin_tab:`tensorflow`

```eval_rst

.. currentmodule:: d2l.torch

```


:end_tab:

## Modèles

```eval_rst 

.. autoclass:: Module
   :members: 

.. autoclass:: LinearRegressionScratch
   :members:

.. autoclass:: LinearRegression
   :members:    

.. autoclass:: Classifier
   :members:

```


## Données

```eval_rst 

.. autoclass:: DataModule
   :members: 

.. autoclass:: SyntheticRegressionData
   :members: 

.. autoclass:: FashionMNIST
   :members: 

```


## Trainer

```eval_rst 

.. autoclass:: Trainer
   :members: 

.. autoclass:: SGD
   :members: 

```


## Utilitaires

```eval_rst 

.. autofunction:: add_to_class

.. autofunction:: cpu

.. autofunction:: gpu

.. autofunction:: num_gpus

.. autoclass:: ProgressBoard
   :members: 

.. autoclass:: HyperParameters
   :members:    

```

