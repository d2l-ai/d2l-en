# Global Model Analysis

Understanding how a model behaves in general is as important as understanding its prediction on a single instance. Global model analyses are beneficial as they can help the stakeholders to get a holistic picture of the model that is making predictions, deliver insights to engineers to vet model's errors, and potentially guide the data collecting process. In what follows, we will experimentally introduce four global model analysis methods: error analysis, ablation study, feature permutation, and partial dependence analysis, using the heart disease dataset. Specifically, we can highlight the landscape of model's predictions across subpopulations using error analysis, sort out the features or model components according to their average contributions using feature ablation or feature permutation, and understand the dependencies between predictions and given features with partial dependence plot.

## Revisiting the MLP Classifier
Now let's train and evaluate the same MLP classifier for heart disease prediction.

```{.python .input}
from d2l import torch as d2l
import pandas as pd
import numpy as np
import torch
import seaborn as sn

data = d2l.HeartDiseaseData()
model = d2l.HeartDiseaseMLP()
trainer = d2l.Trainer(max_epochs=50)
trainer.fit(model, data)
```

This classifier obtains a decent accuracy score on the validation set. Yet, instead of stopping at this point, we can further scrutinize our model by asking: *Q1*: What type of errors have been made by the classification model? *Q2*: How does the model perform on different subgroups? *Q3*: How does each feature contribute to the model performance? *Q4*: What is the relationship between target and the features of interest? We will utilize the three global model analysis methods to address these questions.

## Error Analysis
To answer *Q1* and *Q2*, we need to analyze the errors made in the model. Usual aggregate metrics (e.g., accuracy, mean squared error) for evaluation enable us to understand a model's overall performance but do not reflect the specifics of the errors. For instance, accuracies across different subgroups can be non-uniform, and there might exist cases where the model fails more frequently. To pinpoint such cases, we will conduct error analysis.

Now, we write two functions below, one for error analysis and the other for error map plotting.

```{.python .input}
def error_analysis(model, data, features_of_interest=None):
    X, Y = next(iter(data.val_dataloader()))
    error = pd.DataFrame()
    pred = torch.argmax(model(X), -1)
    error['Actual_Label'] = Y.numpy()
    error['Predicted_Label'] = pred.numpy()
    error['Prediction_Status'] = (pred == Y)
    if features_of_interest is not None:
        for f in features_of_interest:
            error[f] = X[:, data.feat_col.index(f)].numpy()  
    return error

def plot_error_map(x_axis, y_axis, fsize=(2.5, 1.5)):
    d2l.set_figsize(fsize)
    sn.heatmap(pd.crosstab(y_axis, x_axis), annot=True, cmap='Blues', fmt='g')
    d2l.plt.xticks(rotation=30)
    d2l.plt.show()
```

### Confusion Matrix
A confusion matrix provides insights into the error patterns and how the model is confused when making predictions. It can be used to answer *Q1*. Let's plot the confusion matrix below.

```{.python .input}
feat_of_interest = ['sex', 'age', 'cp', 'chol']
error = error_analysis(model, data, feat_of_interest)
plot_error_map(error.Predicted_Label, error.Actual_Label)
```

Confusion matrices can help identify the most promising direction to improve model performance. In particular, among all the misclassifications, we can check if it is false negatives or false positives that account for the majority of the error cases. Spending more effort in inspecting the majority can be more rewarding.

### Errors Distribution across Cohorts
To unveil the erroneous cases across cohorts (to answer *Q2*), we need to stratify them based on the features of interest and specific values of these features. Error stratification can help discover cohorts for which the model underperforms and reveal potential model biases.

In the following, we stratify the erroneous cases across gender groups below (female=0, male=1) to check if gender bias has unintentionally sneaked into the classifier or not.

```{.python .input}
gender_map = { 0: 'Female', 1: 'Male'}
status_map = {False: 'Wrong', True: 'Correct'}
error['sex'] = error['sex'].map(gender_map)
error['Prediction_Status'] = error['Prediction_Status'].map(status_map)
plot_error_map(error['sex'], error['Prediction_Status'])
```

We can also display the error distributions at a finer granularity with feature crossing (e.g., crossing between age groups and gender groups).

```{.python .input}
plot_error_map([pd.cut(error['age'], bins=[0, 45, 65, 120], right=False,
                       labels=['0:45','46:65','66+']), error['sex']], 
               error['Prediction_Status'], fsize=(6, 1.5))
```

Stratifying cases can be laborious sometimes. For example, you may have to manually separate images if you want to check how an image classifier performs on images with different patterns e.g., whether an image is blurry).

## Ablation Study

Ablation study has its roots in the field of experimental neuropsychology, where parts of animals’ brains were removed to study the effect that this had on their behavior. In the context of machine learning, ablation study has been widely adopted to describe the procedure where certain features (components of a model) are removed to understand their influence on the model behavior.  

Details of ablation study vary depending on the application and types of model. Here, we will use feature ablation as an example to demonstrate the process of ablation experiment (We leave model component ablation as an exercise). To goal of feature ablation is to evaluate the influence of features on a model's performance, which can be used to obtain a set of feature importance scores (to answer *Q3*). With a dataset of $p$ features, the procedure goes as follows:
1. Train the model on the entire training set and obtain an accuracy score $s$ (or other scoring metrics) on the validation set.
1. For each feature $i$, $i \in p$, remove it from the training set, then retrain the model and get another score $s_i$ on the validation set.
1. Calculate the global feature importance of feature $i$ using $s-s_i$, which means that the importance scores can be positive, negative, or even zero. In general, the higher the score is, the more important the feature is.

Now, let's implement the above method.

```{.python .input}
def feature_ablation(model, data, feat_of_interest):
    feat_importance = []
    s = model.predict(*next(iter(data.val_dataloader())))
    for i in feat_of_interest:
        used_feat = [feat for feat in data.feat_col if feat != i]
        new_data = d2l.HeartDiseaseData(feat_col=used_feat)
        new_model = d2l.HeartDiseaseMLP()
        new_trainer = d2l.Trainer(max_epochs=50)
        new_trainer.fit(new_model, new_data)
        s_i = new_model.predict(*next(iter(new_data.val_dataloader())))
        d2l.plt.close()
        feat_importance.append(s - s_i)
    return feat_of_interest, feat_importance
```

```{.python .input}
d2l.plot_hbar(*feature_ablation(model, data, ['age', 'cp', 'chol']), 
              xlabel='Feature Importance', fsize=(2, 1.5))
```

## Feature Permutation
The importance of ablation studies can't be stressed enough, but they aren't without shortcomings. One major disadvantage of ablation study is its expensive computational cost due to model retraining. We can avoid this with feature permutation :cite:`Breiman.2001`.

The procedure of feature permutation is similar to feature ablation. The only difference is that, in the second step, we permute (rather than remove) the feature $i$, and then reevaluate (instead of retrain) the model. To permute the feature $i$, we randomly shuffle the values of feature $i$ in the current batch. Feature permutation is more efficient as it does not require model retraining.

```{.python .input}
def feature_permutation(model, data, feat_of_interest):
    feat_importance = []
    X, Y = next(iter(data.val_dataloader()))
    s = model.predict(X, Y)
    for i in feat_of_interest:
        idx = data.feat_col.index(i)
        permuted_X = torch.clone(X)
        permuted_X[:, idx] = permuted_X[:, idx][torch.randperm(X.shape[0])]
        s_i = model.predict(permuted_X, Y)
        feat_importance.append(s - s_i)
    return feat_of_interest, feat_importance
```

```{.python .input}
d2l.plot_hbar(*feature_permutation(model, data, ['age', 'cp', 'chol']),
             xlabel='Feature Importance', fsize=(2, 1.5))
```

Note that it is quite common to get different results using feature ablation and feature permutation. We also have other methods to assess feature importances. For example, some tree-based methods calculate the feature importance using Gini or entropy, but this method is model dependent while feature ablation and feature permutation are model-agnostic.

## Partial Dependence Plot

Feature importance scores are helpful in identifying the main drivers, but they don't provide insights about the relationship between input features and predictions. To answer *Q4*, partial dependence plot (PDP) :cite:`friedman2001greedy` comes to the rescue. PDP is an effective tool to illustrate how the model’s predictions change depending on the values of input features of interest, marginalizing over all other features.  For example, it can show whether the probability of having heart disease increases linearly with age.

Formally, the partial dependence $pd_{x_T}$ between feature $x_T$ and the model's prediction is defined as

$$pd_{x_T}(x_T) = \mathbb{E}_{x_S} [f(x_T, x_S)] \approx \frac{1}{N} \sum_{k=1}^N f(x_T, {x_S}^{(k)}),$$

where $x_T$ is the feature of interest and $x_S$ are all other features; $N$ is the number of examples; and ${x_S}^{(k)}$ is the value of the $k^{\text{th}}$ sample.


The following implementation can output the partial dependence plot with a given target and features of interest (one or two features), and it supports both continuous and categorical features.

```{.python .input}
def partial_dependence_plot(model, data, features, target, 
                            feature_type='continuous'):
    feat_idx_0 = data.feat_col.index(features[0])
    unique_val_0 = np.unique(data.X_test[:, feat_idx_0])
    unique_val_1 = [None]
    if len(features) > 1:
        feat_idx_1 = data.feat_col.index(features[1])
        unique_val_1 = np.unique(data.X_test[:, feat_idx_1])
    Y_hat  = []
    for u_0 in unique_val_0:
        t_0 = []
        for u_1 in unique_val_1:
            t_1 = []
            for x, y in data.train_dataloader():
                x[:,feat_idx_0] = u_0
                if u_1 is not None:
                    x[:,feat_idx_1] = u_1
                t_1.extend(torch.sigmoid(model(x)).detach().numpy()[:, target])
            t_0.append(sum(t_1) / len(t_1))
        Y_hat.append(t_0) if len(features) > 1 else Y_hat.extend(t_0)
    d2l.set_figsize(figsize=(2, 2))
    if len(features) > 1: 
        X, Y = np.meshgrid(unique_val_1, unique_val_0)
        c = d2l.plt.contourf(X, Y, np.array(Y_hat), cmap='GnBu')
        d2l.plt.xlabel(features[1])
        d2l.plt.ylabel(features[0])
        d2l.plt.legend(c.legend_elements()[0], [f'{a:.2f}' for a in c.levels], 
                       loc=(1.05, 0), fontsize='9')
    else:
        d2l.plt.xlabel(features[0])
        d2l.plt.ylabel(f'Probability of target={target}')
        if feature_type.lower() in ['cat', 'categorical']:
            d2l.plt.xticks(unique_val_0)
            d2l.plt.bar(unique_val_0, Y_hat) 
        else:
            d2l.plt.plot(unique_val_0, Y_hat)
```

The following PDP shows how the predicted probability of *target=1* (on the vertical axis) changes by varying the cholesterol level (on the horizontal axis).

```{.python .input}
partial_dependence_plot(model, data, ['chol'], target=1)
```

PDPs with two input features of interest show the interactions among the two features. The PDP below shows how the predicted probability of *target=1* varies with the changes of *thalach* (maximum heart rate achieved) and *age*.

```{.python .input}
partial_dependence_plot(model, data, ['thalach', 'age'], target=1)
```

PDPs are an intuitive way to extract insights from black-box models. Disadvantages of PDPs are: (1) Due to the limits of human perception, we can only show at most three features of interest at a time.  (2) PDP assumes that the features of interest $x_T$ are independent from other features $x_S$, but this assumption is often violated in practice. In the case of correlated features, we will create absurd or unrealistic data points when computing PDP, which may cause mis-interpretations. (3) Another pitfall in PDP is that it does not show the feature distribution, we may overinterpret some regions with very few data points.

## Summary
Global model analysis is important to gain a better understanding of a model's overall behaviors. We can analyze the erroneous cases using error maps, rank the features based on their importance via feature ablation or permutation, and display the partial relationship between predicted targets and features of interest. Methods we covered here are simple yet effective. Later on, we will introduce a few more complex global explanation methods such as SHAP global feature importance, SHAP dependence plot :cite:`Lundberg.Lee.2017`, and submodular pick algorithm :cite:`ribeiro2016should`.


## Exercises
1. Analyze the error distribution across different cholesterol levels.
1. Can you calculate the importance score of the batch norm via ablation study? Hint: You can revise the MLP model.
1. Compute the running time of `feature_ablation` and `feature_permutation`. What can you find from the comparison?
1. Analyze the relationship between *target=1* and resting blood pressure.
