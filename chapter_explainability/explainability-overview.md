# Overview of Model Explainability 

While machine learning has been dramatically improving the state of the art in areas as diverse as machine translation, object recognition, and personalized recommendation, we have also witnessed a shift from simple models to complex models. Simple models such as linear regression and decision trees/lists no longer obtain satisfying performance in many domains, such as on multimodal sources of data (e.g., images, text, videos). 
Indeed, model architectures have been increasingly complicated. For example, CNNs have become deeper, going from just a few layers (e.g., LeNet in :numref:`sec_lenet`) to over a hundred layers (e.g., ResNet in :numref:`sec_resnet`),
and transformers (:numref:`sec_transformer`), although lacking inductive biases from CNNs and RNNs, have become more prevalent. However, these complex models are less understandable to humans. *How to understand what a neuron does in a ResNet?* A better understanding of models, such as how a certain prediction is made, makes us more comfortable before deploying models in high-stakes scenarios.


To begin, let's look at an example of predicting whether a patient has heart disease, given their demographics and medical information. In this example, we will compare a linear classifier with an MLP classifier in terms of prediction accuracy and how easily they can be understood.

```{.python .input}
from d2l import torch as d2l
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn import model_selection
from sklearn import linear_model
```

## A Heart Disease Dataset
Heart disease is one of the major causes of death globally.  Leading risk factors for heart disease include high blood pressure, diabetes, unhealthy cholesterol level, etc. The [heart disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) that we will use is obtained from the UCI machine learning repository. To get started, we will read in the dataset and split it into training and testing sets.

```{.python .input}
class HeartDiseaseData(d2l.DataModule): #@save
    def __init__(self, batch_size=256, test_ratio=0.4, feat_col=None, 
                 target='target'):
        super().__init__()
        self.save_hyperparameters()
        self.df = pd.read_csv(d2l.download(d2l.DATA_URL + 'heart_disease.csv'))
        if feat_col is None:
            self.feat_col = list(set(list(self.df.columns)) - set([target]))
        self.X_train, self.X_test, self.y_train, self.y_test = \
        model_selection.train_test_split(self.df[self.feat_col].values, \
                    self.df[[target]].values, test_size=test_ratio)

    def get_dataloader(self, train):
        if train:
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                torch.from_numpy(self.X_train).type(torch.float), 
                torch.from_numpy(self.y_train).view(-1).type(torch.long)), 
                batch_size=self.batch_size, shuffle=True)
        else:
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                torch.from_numpy(self.X_test).type(torch.float), 
                torch.from_numpy(self.y_test).view(-1).type(torch.long)), 
                batch_size=np.shape(self.X_test)[0], shuffle=False)

    def train_dataloader(self):
        return self.get_dataloader(True)

    def val_dataloader(self):
        return self.get_dataloader(False)

data = HeartDiseaseData()
```

Let's take a look at three examples from the dataset.

```{.python .input}
data.df.sample(n=3, replace=True)
```

This dataset has $13$ feature columns and $1$ *target* column ($1$: has heart disease, $0$: otherwise). The $13$ feature columns are:  *age*, *sex* ($1$: male, $0$: female), *cp* (chest pain type), *trestbps* (resting blood pressure), *chol* (serum cholesterol),  *fbs* (fasting blood sugar > $120$ mg/dl or not), *restecg* (resting electrocardiographic results), *thalach* (maximum heart rate achieved), *exang* (exercise induced angina), *oldpeak* (ST depression induced by exercise), *slope* (the slope of the peak exercise ST segment), *ca* (number of major vessels colored by fluoroscopy), and *thal* (thalassemia). Do not worry if you do not understand some of the medical terms as we will only use a few of them.


## A Linear Classifier for Heart Disease Prediction

Now let's train a linear classifier on the heart disease dataset.

```{.python .input}
lr = linear_model.LogisticRegression(solver='sag', max_iter=20000)
lr.fit(data.X_train, np.squeeze(data.y_train))
print('Validation Accuracy : %.3f'% 
      lr.score(data.X_test, np.squeeze(data.y_test))) 
```

The accuracy is not bad but not that eye-catching. The good thing is that the model is not difficult to understand. For instance, we can align the coefficients with features to inspect their impact.

```{.python .input}
#@save
def plot_feature_importance(names, importance, fsize=(2, 2.5), xlabel=None): 
    d2l.plt.figure(figsize=fsize)
    d2l.plt.barh(range(len(names)), importance)
    d2l.plt.yticks(range(len(names)), names)
    d2l.plt.xlabel(xlabel)
    
plot_feature_importance(data.feat_col, lr.coef_[0])
```

Through this figure, we can roughly know how changes in a feature value affect the model prediction. However, it is inappropriate to directly consider the coefficients of *logistic* regression as feature importance as a sigmoid function is used to convert the output into probabilities. You may do so if the input variables have the same scale and it is a *linear* regression model. In later sections, we will introduce how to interpret logistic regression correctly (e.g., using the odds ratio). The key takeaway here is that it is possible to obtain understandable explanations from linear models.

## An MLP Classifier for Heart Disease Prediction

Now, let's build an MLP classifier for the same dataset. The MLP classifier has five nonlinear layers and each layer has $256$ neurons. We adopt `weight_decay` and `batch_norm` for regularization and `Adam` for network optimization.

```{.python .input}
class HeartDiseaseMLP(d2l.Classifier): #@save
    def __init__(self, num_outputs=2, lr=0.001, wd=1e-6):
        self.save_hyperparameters()
        super(HeartDiseaseMLP, self).__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(), nn.LazyBatchNorm1d(), 
            nn.LazyLinear(256), nn.ReLU(), nn.LazyBatchNorm1d(), 
            nn.LazyLinear(256), nn.ReLU(), nn.LazyBatchNorm1d(), 
            nn.LazyLinear(256), nn.ReLU(), nn.LazyBatchNorm1d(),  
            nn.LazyLinear(num_outputs))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, 
                                weight_decay=self.wd)

    def predict(self, X, Y):
        return self.accuracy(self(X), Y).numpy()
```

We now train and evaluate the MLP classifier.

```{.python .input}
model = HeartDiseaseMLP()
trainer = d2l.Trainer(max_epochs=50)
trainer.fit(model, data)
print('Validation Accuracy : %.3f'% 
      model.predict(*next(iter(data.val_dataloader())))) 
```

As can be seen, the MLP classifier outperforms logistic regression by a wide margin. One possible explanation is that there are some nonlinear patterns in the data that linear models cannot capture. This assumption can be supported by recent research showing that there is a quadratic (U-curve) association between total cholesterol levels and mortality from heart disease :cite:`yi2019total`.

Now let's check to what extent we can understand the MLP classifier. Similarly, we can visualize the weights of the hidden layers of the MLP classifiers to see if we can obtain some insights into the working mechanisms.

```{.python .input}
fig, ax = d2l.plt.subplots(1, 3)
index = 0
for param in model.parameters():
    if param.shape == (256, 256):
        im = ax[index].matshow(param.detach().numpy())
        ax[index].set_axis_off()
        index += 1
fig.colorbar(im, ax=ax.ravel().tolist(), location='bottom', fraction=0.046, pad=0.05)
fig.suptitle('Weights of the hidden layers of the MLP classifier', y=0.01)
d2l.plt.show()
```

Obviously, this figure does not deliver anything that is understandable and useful. Even a simple model like MLP is so difficult to understand, let alone more complex models such as convolutional network networks and Transformer.

## Explainability

*Explainability* in machine learning is the degree to which humans can understand the decisions made by machine learning models. Often, simple models such as linear regression, logistic regression, and decision tree are easy to interpret (or self-explanatory) but fall short in capturing the intricate patterns in data, while complex models (also known as black-box models) sacrifice explainability for better performances. Good explanations are the key to communicating with practitioners and domain experts and enabling a wide deployment of machine learning algorithms. It also provides a new perspective to troubleshoot complex models, speedup debugging processes, bust biases and other potential AI potholes.

Simple models can be valuable in some applications. That is also true in fields other than machine learning. Sometimes, people use easy-to-employ simple models to avoid cognitive overload. For example, emergency medical workers use the [rule of nines](https://en.wikipedia.org/wiki/Wallace_rule_of_nines) :numref:`fig_ruleof9s` to quickly assess a burn's percentage of the total skin;  Recreational divers use a [diving planner](https://en.wikipedia.org/wiki/Recreational_Dive_Planner) to determine how long they can safely stay underwater at a given depth to avoid putting too much nitrogen in their bodies which can cause injury or even death.

![Rule of nines for measuring the burn surface area.](../img/ruleof9s.svg)
:label:`fig_ruleof9s`

Nonetheless, it is undeniable that we have been witnessing a boom of black-box models in the past decade, and many breakthrough results (e.g., image recognition and natural language understanding) are obtained with black-box models. Therefore, we need additional explanation tools when black-box models are necessary, and we want to understand these models. In this chapter, we will introduce some of the popular explanation approaches. Before delving into the details, it is worthwhile to get yourself familar with important terminologies you will encounter.

**Forms of explanations:**
Explanations come in various forms. It can be numerical feature importance values, saliency maps, readable sentences, graphs, or a few representative instances. Explanations ought to be humanly comprehensible and aligned with the vocabulary of target users. That is, explanations that domain experts can understand are not necessarily comprehensible to ordinary users.

**Stakeholders of explanations:**
The stakeholders can be end-users who are directly affected by the AI decisions, AI practitioners who design and implement the AI models, domain experts who will refer to the explanations when making decisions, business owners who are responsible for ensuring AI systems to be aligned with corporate strategies, and governments who should regulate the usage of AI algorithms.

**Inherently interpretable models and post-hoc explanation methods:**
Some machine learning models :cite:`Bishop.2006` such as linear regression, logistic/softmax regression, naive Bayes classifier, k-nearest neighbors, decision trees, decision set, and generalized additive models (GAMs) :cite:`Hastie.Tibshirani.2017` are *inherently explainable* (or *self-explanatory*), which means that they are human-understandable and we can easily figure out how predictions are obtained. When models become more complex and challenging to interpret in itself (deep neural networks, random forests :cite:`Breiman.2001`, support vector machines :cite:`Scholkopf.Smola.2002`, XGBoost :cite:`Chen.Guestrin.2016`, etc.), post-hoc explanations come to the rescue. Figure :numref:`fig_posthoc-xdl` illustrates the pipeline of post-hoc explanation methods. Once a black-box model is trained, we can apply the post-hoc explanantion methods and produce human-understandable explanations.

![A pipline of post-hoc explanation methods.](../img/posthoc-xdl.svg)
:label:`fig_posthoc-xdl`

**Global explanations and local explanations:**
Pertaining the scope of explanations, we can classify explanations into *global explanations* and *local explanations*. Global explanation methods describe the average behavior of a model and connote some sense of understanding of the mechanism by which the model works. A typical global explanation method is the global feature importance plot that displays how much impact on average each feature has on model predictions. On the contrary, local explanation methods are centered around the prediction of each instance and focus on explaining how a specific prediction is obtained for an individual sample. For example, local explanation methods for image classifiers allow us to understand which pixels make an image be classified as a bird. Moreover, local explanations are helpful to vet if individual predictions are being made for the right reasons (e.g., whether gender is heavily used when predicting the eligibility of loan applicants).

**Model-agnostic methods and model-specific methods:**
Based on the applicability of explanation methods, we have *model-agnostic* explanation methods and *model-specific* explanation methods. The former is more flexible and can be applied to any black-box model regardless of its structure, while the latter is limited to specific model classes.



## Pitfalls and Perils

As a fast-growing research area, many unresolved challenges and concerns remain. First and foremost, there are usually no ground-truth labels for evaluating the generated explanations. What the explanation algorithm produce does not necessarily reflect the truth. As such, blindly trusting the explanations and acting on them can lead to dangerous consequences. For the same reason, there is an increasing concern on the efficacy of existing explanation methods, which is sarcastically departing from the original purpose of explanation methods, i.e, increasing the trustworthiness of black-box models.
Furthermore, explanations generated by various methods can disagree with each other drastically, and contradictory explanations are not uncommon :cite:`Krishna.Han.Gu.ea.2022`. As a result, some researchers recommend that inherently interpretable models, rather than black-box models, should be adopted in high-stake scenarios :cite:`Rudin.2019` to avoid potential risks. They also advocate that we should propose more powerful self-explanatory models (e.g., decision lists :cite:`letham2015interpretable`). In the exercise, you may find that decision trees perform very well on the heart disease prediction task.



On the other hand, one may misconsider explainability as causality. Many explanation methods quantify the influence of features or model components by removing or alternating them. It reminds us of counterfactual reasoning, a standard tool for understanding causality, which means reasoning about alternative possibilities for past or future events. For example, would one have heart disease if he had a lower cholesterol level? Explanation methods can explain the causal relationship between inputs and the **predicted** output, but they cannot reflect the real-world causality (i.e., what causes a person to have heart disease). In addition, real-world causality is difficult to discern from observational data :cite:`pearl2009causality`, and sometimes we need to apply interventions, e.g., randomized controlled trials, to measure real-world causality.


## Summary

Model explainability is essential in many domains and the benefits of providing explanations are multifold. There has been an explosion of interest in developing additional explanation methods for black-box models in recent years. We will dive into some of the popular methods and introduce how to apply them to explaining black-box models. Also, bear in mind that the described methods are not de facto solutions, and many critical issues remain underexplored.

## Exercises
1. Can you name some scenarios where explanations are critical?
1. When should we use post-hoc explanation methods rather than self-explanatory methods?
1. What is the performance of *DecisionTreeClassifier* on this dataset?
