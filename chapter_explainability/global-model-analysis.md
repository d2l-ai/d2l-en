# Global Model Analysis

In many cases, we want to vet a model's average behavior instead of that on a particular instance. For example, we may be interested in the error distribution across subpopulations or be keen to dig out the impact of a feature or a learnable component (e.g., weights, networks, and regularizers) on the model's decision process. This section will introduce three model-agnostic approaches: error analysis, ablation study, and partial dependence plot, which can be utilized to get a holistic picture of a model.

In what follows, we will experimentally show how to apply the three model analysis methods via a heart disease prediction task. We import the necessary libraries below.

```{.python .input  n=1}
from d2l import torch as d2l
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sn
from sklearn import model_selection
```

## A Heart Disease Dataset

Heart disease is one of the leading causes of death globally.  Leading risk factors for heart disease include high blood pressure, high cholesterol, diabetes, etc. The [heart disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) that we will use is obtained from the UCI machine learning repository. It has $13$ feature columns and $1$ *target* column ($0$: no heart disease, $1$: has heart disease). 

The $13$ feature columns are:  *age*, *sex* ($1$: male, $0$: female), *cp* (chest pain type. $0$: typical angina, $1$: atypical angina, $2$: non-anginal pain, $3$: asymptomatic), *trestbps* (resting blood pressure), *chol* (serum cholesterol),  *fbs* (fasting blood sugar > $120$ mg/dl. $1$: true, $0$: false), *restecg* (resting electrocardiographic results. $0$: normal, $1$: having ST-T wave abnormality, $2$: showing probable or definite left ventricular hypertrophy by Estes' criteria), *thalach* (maximum heart rate achieved), *exang* (exercise induced angina. $1$: yes, $0$: no), *oldpeak* (ST depression induced by exercise), *slope* (the slope of the peak exercise ST segment.  $0$: upsloping, $1$: flat, $2$: downsloping), *ca* (number of major vessels colored by fluoroscopy), and *thal* (thalassemia, $0$: error, $1$: fixed defect, $2$: normal, $3$: reversible defect). Do not worry if you do not understand some of the terms.

In this task, we will build a predictive model to classify whether a patient has heart disease or not given the aforementioned $13$ features. We use $70\%$ of the examples for training and the remaining $30\%$ for testing.

```{.python .input  n=2}
#@save
class HeartDiseaseData(d2l.DataModule):
    def __init__(self, batch_size=128, test_ratio=0.3, feat_col=None):
        super().__init__()
        self.save_hyperparameters()
        self.df = pd.read_csv(d2l.download(d2l.DATA_URL + 'heart_disease.csv'))
        if feat_col is None:
            self.feat_col = list(set(list(self.df.columns)) - set(['target']))
        self.X_train, self.X_test, self.y_train, self.y_test = \
        model_selection.train_test_split(self.df[self.feat_col].values, \
                    self.df[['target']].values, test_size=test_ratio)

    def get_dataloader(self, train):
        if train:
            X = torch.from_numpy(self.X_train).type(torch.float)
            y = torch.from_numpy(self.y_train).view(-1).type(torch.long)
        else:
            X = torch.from_numpy(self.X_test).type(torch.float)
            y = torch.from_numpy(self.y_test).view(-1).type(torch.long)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            X, y), batch_size=self.batch_size, shuffle=True)

    def train_dataloader(self):
        return self.get_dataloader(True)

    def val_dataloader(self):
        return self.get_dataloader(False)
```

## MLPs for Heart Disease Prediction
Let's construct an MLP classifier for heart disease detection. This model is not inherently interpretable despite its simple structure.

```{.python .input  n=3}
#@save
class HeartDiseaseMLP(d2l.Classifier):
    def __init__(self, num_outputs=2, lr=0.001, wd=1e-6):
        self.save_hyperparameters()
        super(HeartDiseaseMLP, self).__init__()
        self.net = nn.Sequential(nn.LazyLinear(256), nn.LazyBatchNorm1d(), 
                                 nn.ReLU(), nn.LazyLinear(256),
                                 nn.ReLU(), nn.LazyLinear(256),
                                 nn.ReLU(), nn.LazyLinear(256),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, 
                                weight_decay=self.wd)
    
    def predict(self, data, shuffle_x=None):
        Y_hat, Y, X = [], [], []
        for x, y in data:
            if shuffle_x is not None:
                x_col = x[:, shuffle_x]
                x[:, shuffle_x] = x_col[torch.randperm(x.shape[0])]
            Y_hat.append(self(x))
            Y.append(y)
            X.append(x)
        acc = self.accuracy(torch.cat(Y_hat, 0), torch.cat(Y, 0)).numpy()
        return acc, torch.cat(Y_hat, 0), torch.cat(Y, 0), torch.cat(X, 0)
```

Now we train and validate this model.

```{.python .input  n=4}
data = HeartDiseaseData()
model = HeartDiseaseMLP()
trainer = d2l.Trainer(max_epochs=50)
trainer.fit(model, data)
```

As can be seen, our model obtains a decent accuracy score on the validation set. Yet, instead of stopping at this point, we can further scrutinize our model by asking: ***Q1***: What type of errors have been made by the classification model? ***Q2***: How does the model perform on different subgroups? ***Q3***: How does each feature contribute to the model performance? ***Q4***: What is the relationship between target and the features of interest?

## Error Analysis
Aggregate metrics such as accuracy and mean squared error on the entire dataset enable us to understand the overall model performances but do not reflect the specifics of the errors. In particular, the model accuracy may be non-uniform across subgroups, and there might exist cases where the model fails more frequently. We can pinpoint such error cases via error analysis.

We define a *GlobalExplainer* class which requires to be fed with a model and a dataset, and save it for later use.

```{.python .input  n=5}
#@save
class GlobalExplainer: 
    def __init__(self, model, data):
        self.model = model
        self.data = data
```

The function below is used to record the errors and features of interest.

```{.python .input  n=6}
@d2l.add_to_class(GlobalExplainer)
def error_analysis(self, features_of_interest=None):
    acc, pred, Y, X = self.model.predict(self.data.val_dataloader())
    error = pd.DataFrame()
    pred = torch.argmax(pred, -1)
    error['Actual_Label'] = Y.numpy()
    error['Predicted_Label'] = pred.numpy()
    error['Status'] = ['Misclassified' if i == False else 
                       'Correctly_Classified' for i in (pred == Y).numpy()]
    if features_of_interest is not None:
        for f in features_of_interest:
            error[f] = X[:, data.feat_col.index(f)].numpy()  
    return error
```

The following function is used to draw the error maps given the features of interest.

```{.python .input  n=7}
def draw_error_map(x_axis, y_axis):
    d2l.plt.figure(figsize=(2.5, 1))
    sn.heatmap(pd.crosstab(y_axis, x_axis), annot=True, cmap='GnBu', fmt='g')
    d2l.plt.xticks(rotation=30)
    d2l.plt.show()
```

### Confusion Matrix
We can answer ***Q1*** with confusion matrices. Confusion matrices show the inaccuracies patterns and how the model is confused when making predictions. It provides insights into the errors being made and helps identify the most promising direction to improve model performance.

```{.python .input  n=8}
explainer = GlobalExplainer(model, data)
features_of_interest = {'sex', 'age','cp', 'chol', 'fbs'}
error = explainer.error_analysis(features_of_interest)
draw_error_map(error.Predicted_Label, error.Actual_Label)
```

Among all the misclassifications, we can check if it is false negatives or false positives that account for the majority of the error cases. Then, we can spend more effort inspecting the majority of the error cases which might be more rewarding.

### Errors Distribution across Cohorts
The insight from the confusion matrix is helpful, but it does not help us understand the erroneous cases across cohorts. To answer ***Q2***, we need to stratify the erroneous cases further based on the features of interest and specific values of these features. Furthermore, error stratification may reveal possible biases in the model. In the following, error maps are utilized to show details of the model errors and discover cohorts of data for which the model underperforms. 

The following error map stratifies the erroneous cases across different age groups.

```{.python .input  n=9}
draw_error_map(pd.cut(error.age, bins=[0, 30, 45, 60, 120],
                     labels=['0-30','31-45','46-60', '61-older'], 
                     right=False), error.Status)
```

We can display the error distributions at a finer granularity with feature crossing. In this example, we consider two features: fasting blood sugar and sex.

```{.python .input  n=10}
draw_error_map([error.fbs, error.sex], error.Status)
```

The error analyses mentioned above are conducted automatically, but it is also viable to stratify the error cases manually. For example, when analyzing an image classifier, we can manually define the cohorts based on images' patterns (e.g., whether an image is blurry). 


## Ablation Study

Ablation studies refer to removing certain features (or components) of the model and seeing the features' (or components') contribution to the overall model performance. 

### Global Feature Importance via Ablation Study
Feature importance refers to techniques that assign a score to input features based on how useful they are at predicting a target variable. It can be used to answer ***Q3***. Some models, such as random forests provide an innate way of calculating feature importance (e.g., using Gini or entropy) :cite:`Breiman.2001`, but these methods are model dependent.  

With a dataset of $m$ features, the procedure goes like this:
* Train the model on the entire training set and obtain an accuracy score $s$ (or other scoring metrics) on the validation set.
* For each feature $i$, $i \in m$, remove it from the training set, then retrain the model and get another score $s_i$ on the validation set. Alternatively, we can permute the feature $i$ and then re-evaluate the model :cite:`Breiman.2001`. Feature permutation is more efficient as it does not require model retraining.
* Calculate the global feature importance of feature $i$ using $s-s_i$.

Now, let's implement this feature ablation method. The following function provides two ablation modes: *REMOVE* and *PERMUTE*. To permute the feature $i$, we randomly shuffle the values of feature $i$ in the current batch.

```{.python .input  n=11}
@d2l.add_to_class(GlobalExplainer)
def feature_importance(self, features_of_interest, mode='PERMUTE'):
    feat_importance = []
    s = self.model.predict(self.data.val_dataloader())[0]
    for i in features_of_interest:
        if mode == 'PERMUTE':
            s_i = self.model.predict(self.data.val_dataloader(), 
                                    shuffle_x=self.data.feat_col.index(i))[0]
        elif mode == 'REMOVE':
            used_feat = [feat for feat in self.data.feat_col if feat != i]
            new_data = HeartDiseaseData(feat_col=used_feat)
            new_model = HeartDiseaseMLP()
            new_trainer = d2l.Trainer(max_epochs=30)
            new_trainer.fit(new_model, new_data)
            s_i = new_model.predict(new_data.val_dataloader())[0]
            d2l.plt.close()
        feat_importance.append(s - s_i)
    return features_of_interest, feat_importance
```

The following function draws the feature importance scores with a bar chart.

```{.python .input  n=12}
#@save
def draw_feature_importance(features_names, feature_importance):
    d2l.plt.figure(figsize=(2, 2.5))
    num_feat = len(features_names)
    d2l.plt.barh(range(num_feat), feature_importance,)
    d2l.plt.yticks(range(num_feat), features_names)
    d2l.plt.xlabel('Feature Importance')
```

For simplicity, we only output the global importance scores using the *PERMUTE* mode.

```{.python .input  n=13}
draw_feature_importance(*explainer.feature_importance(data.feat_col))
```

Note that the score can be negative, and the higher the score is, the more important the feature is. Similarly, we can remove a component (e.g., a layer or a regularizer) from the model and repeat the same procedure in order to measure the importance of that component. We leave it as an exercise.

Ablation studies are straightforward and useful but they are not without shortcomings. Some ablation methods (e.g., the *REMOVE* mode) can be computationally expensive.  Also, different runs can give different feature importance scores due to the randomness introduced by permutation or retraining.

## Partial Dependence Plot

Feature importance scores are helpful to identifying the main drivers, but they do not provide insights about the relationship between input features and predictions ***Q4***. Partial dependence plot (PDP) :cite:`friedman2001greedy` is an effective tool to illustrate how the model’s predictions change depending on the values of input features of interest, marginalizing over the values of all other input features.  For example, it can show whether the probability of getting heart disease increases linearly with age. The partial dependence $pd_{X_T}$ between feature $X_T$ and the model's prediction is defined as
$$
pd_{X_T}(x_T) = \mathbb{E}_{X_S} [f(x_T, X_S)] \approx \frac{1}{N} \sum_{k=1}^N f(x_T, {x_S}^{(k)}),
$$
where $X_T$ are the features of interest and $X_S$ are other features; $N$ is the number of examples; and ${x_S}^{(k)}$ is the value of the $k^{\text{th}}$ sample for the features in $X_S$.

The following function draws the partial dependence plot with a given target and features of interest (one or two features). The *feature_type* can be *NUMERICAL* or *CATEGORICAL*.

```{.python .input  n=14}
@d2l.add_to_class(GlobalExplainer)
def PartialDependencePlot(self, features, target, feature_type='NUMERICAL'):
    feat_idx_0 = self.data.feat_col.index(features[0])
    unique_val_0 = np.unique(self.data.X_test[:, feat_idx_0])
    unique_val_1 = [None]
    if len(features) > 1:
        feat_idx_1 = self.data.feat_col.index(features[1])
        unique_val_1 = np.unique(self.data.X_test[:, feat_idx_1])
    Y_hat  = []
    for u_0 in unique_val_0:
        temp_0 = []
        for u_1 in unique_val_1:
            temp_1 = []
            for x, y in self.data.val_dataloader():
                x[:,feat_idx_0] = u_0
                if u_1 is not None:
                    x[:,feat_idx_1] = u_1
                temp_1.extend(torch.sigmoid(self.model(x)).detach() \
                              .numpy()[:, target])
            temp_0.append(sum(temp_1) / len(temp_1))
        Y_hat.append(temp_0) if len(features) > 1 else Y_hat.extend(temp_0)
    d2l.plt.figure(figsize=(2, 2))
    if len(features) > 1: 
        X, Y = np.meshgrid(unique_val_1, unique_val_0)
        c = d2l.plt.contourf(X, Y, np.array(Y_hat), cmap='GnBu')
        d2l.plt.xlabel(features[1])
        d2l.plt.ylabel(features[0])
        d2l.plt.legend(c.legend_elements()[0], [f'{a:.2f}' for a in c.levels], 
                       loc=(1.05, 0), fontsize='9')
    else:
        d2l.plt.xlabel(features[0])
        d2l.plt.ylabel('Probability of target=' + str(target))
        if feature_type.lower() in ['cat', 'categorical']:
            d2l.plt.xticks(unique_val_0)
            d2l.plt.bar(unique_val_0, Y_hat) 
        else:
            d2l.plt.plot(unique_val_0, Y_hat)
```

The following partial dependence plot shows how the predicted probability of target=1 (on the vertical axis) changes as we increase the cholesterol level (on the horizontal axis).

```{.python .input  n=15}
explainer.PartialDependencePlot(['chol'], 1, feature_type='NUMERICAL')
```

PDPs with two input features of interest show the interactions among the two features. The PDP below shows how the predicted probability of target=1 varies with the changes of *thalach* (maximum heart rate achieved) and *age*.

```{.python .input  n=16}
explainer.PartialDependencePlot(['thalach', 'age'], 1)
```

PDPs are an intuitive way to extract insights from black-box models. Disadvantages of PDPs are: (1) Due to the limits of human perception, we can only show at most three features of interest at a time.  (2) PDP assumes that the features of interest $X_T$ are independent from other features $X_S$, but this assumption is often violated in practice. In the case of correlated features, we will create absurd or unrealistic data points when computing PDP, which may cause mis-interpretations. (3) Another pitfall in PDP is that it does not show the feature distribution, we may overinterpret some regions with very few data points (e.g., *chol*>400).

## Summary
* Error analysis helps us get insights into the erroneous cases.
* Feature importance can be obtained via ablation studies.
* Partial dependence plots can show the relationship between targets and features of interest.

## Exercises
1. Analyze the error distribution across different cholesterol levels. 
1. Can you calculate the importance score of the batch norm via ablation study? Hint: You may need to revise the MLP model.
1. Compute the global feature importance using *REMOVE* mode and compare the results and runtime with the *PERMUTE* mode. What can you find from the comparison?
1. Analyze the relationship between *target* and resting blood pressure.
