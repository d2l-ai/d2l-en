## Which metrics are important?


MS: I think this is good for now, but we may want to check whether some basics here are already well explained in
early chapters, and if so, just refer to them.

### Objective Function

Arguably the most common way to estimate the validation performance of a machine learning algorithm is to compute its error (e.g classification error) on a hold out validation set. We cannot use the training loss to optimize the hyperparameters, as this would lead to overfitting. Unfortunately, in case of small datasets we often do not have access to a sufficient large validation dataset. In this case we can apply $k$-fold cross validation and use the average validation loss across all folds as metric to optimize. However, at least in the standard form of HPO, this makes the optimization process $k$ times slower. 

We can generalize the definition of HPO in order to deal with multiple objectives $f_0, ... f_k$ at the same time. For example, we might not only be interested in optimize the validation performance, but also, for example, the number of parameters:

```{.python .input  n=8}
def multi_objective(config, max_epochs = 10):
    batch_size = config['batch_size']
    lr = config['learning_rate']

    model = AlexNet(lr=lr)
    trainer = d2l.Trainer(max_epochs=max_epochs, num_gpus=0)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(224, 224))
    trainer.fit(model=model, data=data)
            
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    validation_error = trainer.evaluate()

    return validation_error, num_params
```

However, this means we will not have a single $\mathbf{x}$ anymore that optimizes all objective functions at the same time. We can see this in the figure below: Each points marked in red represents a hyperparameter configuration that dominates all other sampled configurations in one of the objectives. This set of points is called the Parteto set.

TODO: this figure will be replaces by something that matches the code above. The message of the plot is not changing though.

![Pareto front of two objectives](img/pareto_front.png)
:width:`400px`
:label:`pareto_front`

### Cost

Another relevant metric is the cost of evaluating $f(\mathbf{x})$ at a configuration $\mathbf{x}$. Different to, for example, the validation error, this metric is not a function of the final trained model, but a measure of training wall-clock time. For example, if we tune the number of layers or units per layer, larger networks are slower
to train than smaller ones, but potentially lead to a lower validation error. In our runnning example, training time does not depend on the `learning_rate` but varies in general with the `batch_size`. Counting cost in terms of wall-clock time is more relevant in practice than counting the number of evaluations. Some HPO algorithms explicitly model training cost and take it into
account for making decisions.

```{.python .input  n=9}
import time

def objective_function_with_cost(config, max_epochs=10):
    start_time = time.time()
    validation_error = objective(config, max_epochs)
    return validation_error, time.time() - start_time
```

### Constraints

In many scenarios we are not just interested in finding $\mathbf{x}_{\star}$, but a hyperparameter configuration that additionally full fills certain constraints. More formally, we seek to find $\mathbf{x}_{\star} \in argmin_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$ s.t $c_1(\mathbf{x}) > 0, ..., c_m(\mathbf{x}) > 0$. Typical constraints could be, for example, the memory consumption of $\mathbf{x}$ or fairness constraints.
