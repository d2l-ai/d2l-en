# Global Model Analysis

In many cases, we want to vet the model's overall behavior instead of that on a particular instance. For example, we may be concerned about the error distributions or be keen to dig out the impact of a feature or a learnable component (e.g., weights, networks, and regularizers) on the model's decision process in order to build more accurate and generalizable machine learning models. This section will introduce two typical global model analysis approaches, error analysis and ablation study, which enable us to get a big picture of the entire model.

In what follows, we will empirically show how to conduct global model analysis via a loan defaulter prediction task.  We import the necessary libraries below.

```{.python .input  n=6}
from d2l import torch as d2l
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sn
from sklearn import model_selection
```

## A Loan Dataset

In this task, we will build a predictive model to classify whether a borrower will be a loan defaulter or not. A loan is a sum of money that individuals, companies, or organizations borrow from banks or other financial institutions. It is a core business in many financial institutions, and it can help the borrowers overcome their financial constraints and financially manage events such as paying tuition or buying houses. Although being profitable and beneficial for both lenders and borrowers, it carries risks, and loan default is one of the significant risks. Loan default is when a borrower breaks his/her original agreement with a creditor or lender by discontinuing payments. Loan default will lead to financial loss, so it is crucial to assess borrowers' credit risk to see if they will repay the loan in due time.


Now let's get a loan dataset for loan defaulter prediction. This dataset is a subset of a Kaggle competition: [lending club](https://www.kaggle.com/datasets/wordsforthewise/lending-club). We have deleted unnecessary columns, imputed missing values, removed duplicate entries, and resampled the imbalanced class. The resulting dataset consists of around ten thousand examples. There are $12$ feature columns including *loan_amnt* (loan amount), *term* (loan term), *int_rate* (interest rate), *annual_inc* (annual income).  The homeownership type is represented with four dummy columns: *MORTGAGE*, *OWN*, *RENT*, and *OTHER*. The *loan_paid* column is the target we are going to predict and it can be either fully-paid (non-defaulter) or charge-off (defaulter). Other columns are described in the Table below. Note that [FICO scores](https://www.fico.com/en/products/fico-score) represent how credit-worthy a person is. 


| Column                   | Description                                                                  |
|--------------------------|------------------------------------------------------------------------------|
| *fico_range_high*          | The upper boundary range the borrower's FICO at loan origination belongs to. |
| *last_fico_range_high*    | The upper boundary range the borrower's last FICO pulled belongs to.         |
| *last_pymnt_amnt*        | The last total payment amount received.                                           |
| *chargeoff_within_12_mths* | The number of charge-offs within $12$ months.                               |


We split the dataset and use $70\%$ of the examples for training and the remaining $30\%$ for testing.

```{.python .input  n=9}
#@save
class LoanData(d2l.DataModule):
    def __init__(self, url=d2l.DATA_URL + 'loan.csv', batch_size=64,
                 test_ratio=0.3, feat_col=None, target_col=['loan_paid']):
        super().__init__()
        self.save_hyperparameters()
        self.data = pd.read_csv(d2l.download(url))
        if feat_col is None:
            self.feat_col = list(set(list(self.data.columns))-set(target_col))
        self.X_train, self.X_test, self.y_train, self.y_test = \
        model_selection.train_test_split(self.data[self.feat_col].values, \
                        self.data[target_col].values, test_size=test_ratio)

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
    
data = LoanData(url=d2l.DATA_URL + 'loan.csv', batch_size=64, test_ratio=0.3)
```

## MLPs for Loan Defaulter Prediction
We construct a multilayer perceptron with a few fully-connected layers for loan defaulter prediction. This model is not inherently interpretable despite its simple structure.

```{.python .input  n=3}
#@save
class MlpLoan(d2l.Classifier):
    def __init__(self, num_outputs, lr, wd):
        self.save_hyperparameters()
        super(MlpLoan, self).__init__()
        self.net = nn.Sequential(nn.LazyLinear(32), nn.BatchNorm1d(32), 
                                 nn.ReLU(), nn.LazyLinear(32), 
                                 nn.ReLU(), nn.LazyLinear(32), 
                                 nn.ReLU(), nn.LazyLinear(num_outputs))

    def forward(self, x):
        return self.net(x)
    
    def loss(self, y_hat, y):
        fn = nn.CrossEntropyLoss()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.wd)
    
    def predict(self, data, shuffle_x=None):
        Y_hat, Y, X = [], [], []
        for x, y in data:
            if shuffle_x is not None:
                x_col = x[:,shuffle_x]
                x[:, shuffle_x] = x_col[torch.randperm(x.shape[0])]
            Y_hat.append(self(x))
            Y.append(y)
            X.append(x)
        acc = self.accuracy(torch.cat(Y_hat, 0), torch.cat(Y, 0)).numpy()
        return acc, torch.cat(Y_hat, 0), torch.cat(Y, 0), torch.cat(X, 0)
```

Let's train and validate this model.

```{.python .input  n=2}
model = MlpLoan(num_outputs=2, lr=0.001, wd=1e-5)
trainer = d2l.Trainer(max_epochs=20)
trainer.fit(model, data)
print("Validation accuracy is:", model.predict(data.val_dataloader())[0])
```

As can be seen, our model obtains a decent accuracy score on the validation set. Yet, instead of stopping at this point, we can further scrutinize our model by asking: (1) What type of errors have been made by the classification model? (2) How does the model perform on different groups of loan borrowers? (3) How does each feature contribute to the model performance?

## Error Analysis
As what we have done, aggregate metrics such as accuracy and mean squared error on the entire dataset are usually adopted to assess the effectiveness of machine learning models. These metrics help us understand the overall model performances but do not reflect the specifics of the errors. In particular, the model accuracy may be non-uniform across subgroups of data, and there might exist cases where the model fails more frequently. Pinpointing such cases via error analysis is helpful in model debugging and the detection of root causes of errors. 

In the following, we will demonstrate how to identify the error distributions across different cohorts via error analysis.  We define a *GlobalExplainer* class which requires to be fed with a black-box model and a dataset, and save it for later use.

```{.python .input  n=5}
#@save
class GlobalExplainer: 
    def __init__(self, black_box_model, data):
        self.black_box_model = black_box_model
        self.data = data
        
explainer = GlobalExplainer(model, data)
```

The function below is used to record the errors and features of interest.

```{.python .input  n=3}
@d2l.add_to_class(GlobalExplainer)
def error_analysis(self, feat_of_interest=None):
    acc, pred, Y, X = self.black_box_model.predict(self.data.val_dataloader())
    error = pd.DataFrame()
    pred = torch.argmax(pred, -1)
    error['Actual_Label'] = ["Fully-Paid" if i==1 else "Charge-Off" 
                                    for i in Y.cpu().detach().numpy()]
    error['Predicted_Label'] = ["Fully-Paid" if i==1 else "Charge-Off" 
                                  for i in pred.cpu().detach().numpy()]
    error['Status'] = ["Misclassified" if i==False else "Correctly_Classified" 
                                for i in (pred == Y).cpu().detach().numpy()]
    if feat_of_interest is not None:
        for key, value in feat_of_interest.items():
            error[key] = X[:, value].cpu().detach().numpy()  
   
    return error
```

The following function is used to draw the error maps given the features of interest.

```{.python .input  n=7}
@d2l.add_to_class(GlobalExplainer)
def draw_heat_map(self, x_axis, y_axis):
    d2l.plt.figure(figsize = (3,1))
    sn.heatmap(pd.crosstab(y_axis, x_axis), annot=True, cmap="GnBu", fmt='g')
    d2l.plt.xticks(rotation=20)
    d2l.plt.show()
```

### Confusion Matrix
We can answer the first question through a confusion matrix. Confusion matrices show the inaccuracies patterns and how the model is confused when making predictions. It can provide insights into the errors being made and help identify the most promising direction to improve model performance.

```{.python .input  n=8}
feat_of_interest = {"loan_amnt": data.feat_col.index("loan_amnt"),
               "annual_inc": data.feat_col.index("annual_inc"),
               "term": data.feat_col.index("term"),
               "OWN": data.feat_col.index("OWN") }
error = explainer.error_analysis(feat_of_interest)
explainer.draw_heat_map(error.Predicted_Label, error.Actual_Label)
```

Among all the misclassifications, we can check if it is false negatives or false positives that account for the majority of the error cases. Then, we can spend more effort inspecting the majority of the error cases as improving the prediction on them may be more rewarding.

### Errors Distribution across Cohorts
The insight from the confusion matrix is helpful, but it does not help us understand the erroneous cases across cohorts (e.g., borrowers with different annual income). To this end, we need to stratify the erroneous cases further based on the features of interest and specific values of these features. In particular, error maps are utilized to show details of the error specifies and discover cohorts of data for which the model underperforms.

The following line can diagnose the error distribution across cohorts of different loan terms.

```{.python .input  n=9}
explainer.draw_heat_map(error.term, error.Status)
```

From this error map, we observe the difference of error rate on different loan term (36 months vs. 60 months). Evidently, the model makes less mistakes on long-term loans. If the feature of interest is sensitive (e.g., gender), this kind of disparity will lead to unacceptable discrimination and unfaireness. Error stratification can keep us aware of possible model biases.

Similarly, we can also stratify the erroneous cases based on the annual income by classifying the income into different levels.

```{.python .input  n=10}
explainer.draw_heat_map(pd.qcut(error.annual_inc, q=3), error.Status)
```

We can even display the error distributions at a finer granularity with feature crossing. In this example, we consider two features: the term of loan and whether the borrower fully owns their house (OWN: 1=Yes, 0=No).

```{.python .input  n=11}
explainer.draw_heat_map([error.term, error.OWN], error.Status)
```

The error analyses mentioned above are conducted automatically, but it is also viable to stratify the error cases manually. For example, when analyzing an image classifier, we can manually define the cohorts based on images' patterns (e.g., whether an image is blurry). 


## Ablation Study

Ablation studies refer to removing certain features (or components) of the model and seeing the features' (or components') contribution to the overall model performance. Some models, such as random forests provide an innate way of calculating feature importance (e.g., using Gini or entropy) :cite:`Breiman.2001`, but these methods are model dependent. Ablation studies are model-agnostic and can be applied to any kind of model.


### Global Feature Importance via Ablation Study
Feature importance refers to techniques that assign a score to input features based on how useful they are at predicting a target variable. With a dataset of $m$ features, the procedure goes like this:

* Train the model on the entire training set and obtain an accuracy score $s$ (or other scoring metric) on the validation set.
* For each feature $i$, $i \in m$, remove it from the training set, then retrain the model and get another score $s_i$ on the validation set. Alternatively, we can permute the feature and then re-evaluate the model :cite:`Breiman.2001`. Feature permutation is more efficient as it does not require model retraining.
* Calculate the global feature importance score with $s-s_i$.

Now, let's implement this feature ablation method. The following function provides two ablation modes: *REMOVAL* and *SHUFFLE*. To permute the feature $i$, we randomly shuffle the values of feature $i$ in the current batch.

```{.python .input  n=12}
@d2l.add_to_class(GlobalExplainer)
def feature_importance(self, feat_of_interest, mode="SHUFFLE"):
    feature_importance = {}
    s, _, _, _ = self.black_box_model.predict(self.data.val_dataloader())
    for i in feat_of_interest:
        if mode == "SHUFFLE":
            s_i, _, _, _ = self.black_box_model.predict(
                self.data.val_dataloader(), shuffle_x=self.data.feat_col.index(i))
        elif mode == "REMOVAL":
            used_feat = [feat for key, feat in enumerate(self.data.feat_col) 
                         if feat!=i]
            new_data = LoanData(feat_col=used_feat)
            new_model = MlpLoan(num_outputs=2, lr=0.001, wd=1e-5)
            trainer = d2l.Trainer(max_epochs=20)
            trainer.fit(new_model, new_data)
            s_i, pred, Y, X = new_model.predict(
                new_data.val_dataloader())
        feature_importance[i] = (s-s_i) 
        d2l.plt.close()
        
    d2l.plt.figure(figsize = (3,2))
    d2l.plt.barh(range(len(feat_of_interest)), feature_importance.values())
    d2l.plt.yticks(range(len(feat_of_interest)), list(feature_importance.keys()))
    d2l.plt.xlabel('Global Feature Importance score using '+mode)
    d2l.plt.show()
```

The following code outputs the importance of five features: loan amount, loan term, interest rate, annual income, and the number of charge-offs within 12 months.

```{.python .input  n=13}
feat_2_analyze = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 
                  'chargeoff_within_12_mths']
explainer.feature_importance(feat_2_analyze, mode="REMOVAL")
explainer.feature_importance(feat_2_analyze, mode="SHUFFLE")
```

Note that the score can be negative, and the higher the score is, the more important the feature is. Via this ablative study, we can know which feature is the most predictive. 

Similarly, we can remove a component (e.g., a layer or a regularizer) from the model and repeat the same procedure in order to measure the importance of that component. We leave it as an exercise.

Ablation studies are straightforward and useful but they are not without shortcomings. As you may notice, the relative feature importance can be different with different ablation modes (i.e., REMOVAL and SHUFFLE). This phenomenon is quite common in explanation methods. Also, some ablation studies can be computationally expensive. Although permutation approaches can alleviate this problem, additional randomness will be introduced, that is, different runs might give different relevant feature importance.

## Summary
* Error analysis helps us get insight into the erroneous cases of the model.
* Feature importance can be obtained via ablation studies regardless of the model type.

## Exercises
1. Analyze the error distribution across different groups of the loan amount. Hint: You may need to rewrite the MlpLoan model.
2. Can you calculate the importance score of the batch norm via ablation study?
