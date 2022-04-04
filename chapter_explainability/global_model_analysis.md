# Global Model Analysis

In many cases, we want to vet the model's overall behavior instead of that on a particular instance. For example, we may be concerned about the error distributions and are keen to dig out the impact of a feature and a learnable component (e.g., weights, networks, regularizers, etc.) on the model's decision process. This section will introduce two typical global model analysis approaches, including error analysis and ablation study, which enable us to understand the entire model in depth.


## Error Analysis
Aggregate metrics such as accuracy, AUC, and mean absolute errors on the entire dataset are usually adopted to assess the effectiveness of machine learning models. These metrics help us understand the overall model performances but do not reflect the specifics of the errors. For example, the model accuracy may be non-uniform across subgroups of data, and there might exist cases where the model fails more frequently. Pinpointing such cases via error analysis is helpful in model debugging and detection of root causes of errors. Furthermore, by doing so,  we can build more accurate, generalizable, and robust machine learning models.

In what follows, we will empirically show how we can identify the errors distributions across different cohorts at different levels of granularity through a loan defaulter prediction task. We import the necessary libraries below.

```{.python .input  n=15}
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import seaborn as sn
from torch.autograd import Variable
from d2l import torch as d2l
```

In this task, we will build a predictive model to classify whether a borrower will be a loan defaulter or not. A loan is a sum of money that individuals, companies, or organizations borrow from banks or other financial institutions. It is a core business in many financial institutions, and it can help the borrowers overcome their financial constraints and financially manage events such as paying tuition, buying houses, etc. Although being profitable and beneficial for both lenders and borrowers, it carries risks, and loan default is one of the significant risks. Loan default is when a borrower breaks his/her original agreement with a creditor or lender by discontinuing payments. Loan default will lead to financial loss, and it is crucial to assess borrowers' credit risk to see if they will repay the loan in due time.

### Getting and splitting the data
Now let's get a loan dataset for loan defaulter prediction. This dataset is a subset of a [Kaggle competition](https://www.kaggle.com/datasets/wordsforthewise/lending-club). We have deleted unnecessary columns, imputed missing values, removed duplicate entries, and resampled the imbalanced class. The resulting dataset consists of around ten thousand examples.

```{.python .input  n=17}
df = pd.read_csv(d2l.download(d2l.DATA_URL + 'loan.csv'))
```

There are $12$ feature columns including *loan_amnt* (loan amount), *term* (loan term), *int_rate* (interest rate), *annual_inc* (annual income).  The homeownership type is the four columns: ' MORTGAGE', 'OTHER', 'OWN', and 'RENT'. Other columns are described in the Table below. The *loan_paid* column is the target we are going to predict and it can be either fully-paid (non-defaulter) or charge-off (defaulter).

| Column                   | Description                                                                  |
|--------------------------|------------------------------------------------------------------------------|
| fico_range_high          | The upper boundary range the borrower's FICO at loan origination belongs to. |
| last_fico_range_high     | The upper boundary range the borrower's last FICO pulled belongs to.         |
| last_pymnt_amnt          | Last total payment amount received.                                           |
| chargeoff_within_12_mths | Number of charge-offs within 12 months.                               |


We then split the data and use $70\%$ of the examples for training and the remaining $30\%$ for testing.

```{.python .input  n=3}
def data_split_load(df, training_ratio, in_feat, target_colum):
    train = df.sample(frac=training_ratio,random_state=30) 
    test = df.drop(train.index)
    X_train, y_train = train[in_feat].values, train[target_colum].values
    X_test, y_test = test[in_feat].values, test[target_colum].values
    X_train = torch.from_numpy(X_train).type(torch.float)
    y_train = torch.from_numpy(y_train).view(-1).type(torch.long)
    X_test = torch.from_numpy(X_test).type(torch.float)
    y_test = torch.from_numpy(y_test).view(-1).type(torch.long)
    train_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        X_train,y_train), batch_size=64, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        X_test,y_test), batch_size=64, shuffle=False)
    return train_iter, test_iter
target_column = ['loan_paid']
feat_columns = list(set(list(df.columns))-set(target_column))
train_iter, test_iter = data_split_load(df, 0.7, feat_columns, target_column)
```

### Multilayer perceptrons for loan defaulter prediction
We construct a multilayer perceptron with a few fully-connected layers for loan defaulter prediction.  Even though the model structure is simple, it is a black-box model and not inherently interpretable.

```{.python .input  n=4}
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.bn = nn.BatchNorm1d(32)
        self.out_layer = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.fc3(self.fc2(torch.relu(self.bn(self.fc1(x)))))
        x = self.out_layer(x)
        return x
```

Let's train and test the model.

```{.python .input  n=5}
devices = d2l.try_gpu()
lr, num_epochs = 0.001, 20
net = MLP(input_dim=len(feat_columns), output_dim=2)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, devices, 'Adam')
```

As can be seen, our model obtains a decent accuracy score. Yet, instead of stopping at this point, we can further scrutinize our model by asking: (1) What type of errors have been made by the classification model? (2) How the model performs on different groups of loan borrowers? (3) Which feature is the more important to the model performance? 
To answer these questions, we will utilize the following function.

```{.python .input  n=6}
def error_analysis(net, data_iter, device, feat_2_analyze=None, shuffleX=None):
    net.eval()
    num_examples, correct, error_analysis, pred_label = 0, 0, [], []
    with torch.no_grad():
        for X, y in data_iter:
            temp_df = pd.DataFrame()
            X, y = X.to(device), y.to(device)
            if shuffleX is not None:
                X_col = X[:,shuffleX]
                X[:, shuffleX] = X_col[torch.randperm(X.shape[0])]
            output = torch.argmax(net(X), -1)
            temp_df['Actual_Label'] = ["Fully-Paid" if i==1 else "Charge-Off" 
                                    for i in y.cpu().detach().numpy()]
            temp_df['Predicted_Label'] = ["Fully-Paid" if i==1 else "Charge-Off" 
                                  for i in output.cpu().detach().numpy()]
            temp_df['Status'] = ["Misclassified" if i==False else "Correctly_Classified" 
                                for i in (output == y).cpu().detach().numpy()]
            if feat_2_analyze is not None:
                for key, value in feat_2_analyze.items():
                    temp_df[key] = X[:, value].cpu().detach().numpy()   
            num_examples += len(y)
            correct += (output == y).sum()
            error_analysis.append(temp_df)
        error_analysis = pd.concat(error_analysis, ignore_index=True)
    return 1.0 * correct/num_examples, error_analysis
```

### Confusion matrix
We can answer the first question roughly with a confusion matrix. Confusion matrices show how the model is confused when making predictions. It can provide insights into the errors being made and help identify the most promising direction to improve model performance.

```{.python .input  n=7}
# The features groups of interest.
feat_2_analyze = {"loan_amnt": feat_columns.index("loan_amnt"),
               "annual_inc": feat_columns.index("annual_inc"),
               "term": feat_columns.index("term"),
               "OWN": feat_columns.index("OWN") }

accuracy, err_analysis = error_analysis(net, test_iter, devices, feat_2_analyze)

confusion_matrix = pd.crosstab(err_analysis['Actual_Label'], err_analysis['Predicted_Label'])
d2l.plt.figure(figsize = (3,1))
sn.heatmap(confusion_matrix, annot=True, cmap="GnBu", fmt='g')
d2l.plt.show()
```

This confusion matrix gives us a sense of inaccuracies patterns. Among all the misclassifications, we can check if it is false negatives or false positives which account for the majority and then spend more effort improving the majority of error cases.

### Errors distribution across different cohorts
The insights from the confusion matrix are helpful, but it does not help us understand the erroneous cases across cohorts (e.g., people with different annual income). To answer this question, we need to stratify further the erroneous cases based on the feature of interest and specific values of these features. In particular, error maps are utilized to show details on the error specifies and help discover cohorts of data for which the model underperforms.

The following code will help us diagnose the error distribution across different loan term groups.

```{.python .input  n=8}
d2l.plt.figure(figsize = (3,1))
sn.heatmap(pd.crosstab(err_analysis.Status, err_analysis.term),
            cmap="GnBu", annot=True, fmt='g')
d2l.plt.show()
```

This error map shows that the error rate on short-term loans (36 months) is larger than that on longer-term loans (60 months). This kind of model bias is commonplace due to skewed datasets, and the error stratification enables us to be more aware of them.

Similarly, we can also stratify the erroneous cases via borrowers' income levels. We split the annual income into three groups and check the error distributions.

```{.python .input  n=9}
d2l.plt.figure(figsize = (3,1))
sn.heatmap(pd.crosstab(err_analysis['Status'],pd.qcut(err_analysis['annual_inc'], q=3)
                       ), cmap="GnBu", annot=True, cbar=False, fmt='g')
d2l.plt.xticks(rotation=30)
d2l.plt.show()
```

We can even diagnose the error distributions across two or more features.

```{.python .input  n=10}
d2l.plt.figure(figsize = (3,1))
sn.heatmap(pd.crosstab(err_analysis.Status, [err_analysis.term, err_analysis.OWN]),
            cmap="GnBu", annot=True, fmt='g')
d2l.plt.xticks(rotation=25)
d2l.plt.show()
```

In this example, we consider the loan term and whether the borrower wholly owns their house (OWN: 1=Yes, 0=No). It shows how we can stratify the erroneous cases at a finer granularity.

The error analyses mentioned above are conducted automatically, but it is also viable to stratify the error cases manually. For example, when analyzing an image classifier, you can manually define the cohorts based on images' patterns (e.g., whether an image is blurry). Then, once the errors have been identified, we can explore and debug the respective features or cohorts in a more detailed way.

## Ablation Study

Ablation studies refer to removing certain features (or components) of the model and seeing the features' (or components) contribution to the overall model performance. Some models, such as random forests, xgboost provide an innate way of calculating feature importance (e.g., using Gini or entropy), but these methods are model dependent. Ablation studies are model-agnostic and can be applied to any kind of model.


### Feature ablation
Feature importance refers to techniques that assign a score to input features based on how useful they are at predicting a target variable. With a dataset of $m$ features, the procedure goes like this:

* Train the model on the entire training set and obtain an accuracy score $s$ (or other scoring metric) on the test set.
* For each feature $i$, $i \in m$, remove it from the training set, then train the model and get another score $s_i$ on the test set. Alternatively, you can permute the feature and then reevaluate the model. Permuting the features does not require model retraining, which is more efficient. More permutation strategies can be found in :cite:`fisher2019all`.
* Calculate the importance score by ranking the features by $s-s_i$.

Now, let's implement this feature ablation method.

```{.python .input  n=11}
def feat_importance_ablation(feat_2_analyze, feat_columns, net, train_iter, 
                             test_iter, devices, s, mode="SHUFFLE"):
    feat_importance = {}
    for i in feat_2_analyze:
        if mode == "SHUFFLE":
            s_i, _ = error_analysis(net, test_iter, devices, shuffleX=feat_columns.index(i))
        elif mode == "REMOVAL":
            used_feat = [feat for key, feat in enumerate(feat_columns) if feat!=i]
            train_iter, test_iter = data_split_load(df, 0.7, used_feat, target_column)
            net = MLP(input_dim=len(used_feat), output_dim=2)
            d2l.train_ch6(net, train_iter, test_iter, 20, lr, devices, 'Adam')
            s_i, _ = error_analysis(net, test_iter, devices)
        feat_importance[i] = (s-s_i).cpu().detach().numpy()  
        d2l.plt.close()
        
    d2l.plt.figure(figsize = (3,2))
    d2l.plt.barh(range(len(feat_importance)), feat_importance.values())
    d2l.plt.yticks(range(len(feat_importance)), list(feat_importance.keys()))
    d2l.plt.xlabel('Importance score with '+mode)
    d2l.plt.show()
```

This function provides two ablation modes: *REMOVAL* and *SHUFFLE*. To permute the feature $i$, we randomly shuffle the values of feature $i$ in each batch. For example, suppose we are interested in comparing the importance of five features, including loan amount, loan term, interest rate, annual income, and the number of charge-offs within 12 months.

Here we test both modes.

```{.python .input  n=12}
feat_2_analyze = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'chargeoff_within_12_mths']
feat_importance_ablation(feat_2_analyze, feat_columns, net, train_iter, 
                             test_iter, devices, accuracy, mode="REMOVAL")
feat_importance_ablation(feat_2_analyze, feat_columns, net, train_iter, 
                             test_iter, devices, accuracy, mode="SHUFFLE")
```

Via this ablative study, we will know which feature is the most predictive among the five features. We also notice that different feature ablation methods can lead to different results, consistent with the findings in :cite:`Krishna.Han.Gu.ea.202`. Note that the score can be negative, and the higher the score is, the more important the feature is.

Similarly, we can also remove components (e.g., a layer, an activation function, etc.) from the model and repeat the same procedure in order to get the importance score of model components. We leave it as an exercise.


Ablation studies are straightforward and useful but they are not without shortcomings. One major issue is the high computation cost of retraining a model. Even though, for feature ablations, we can avoid model retraining with feature permutations, it can introduce additional randomness when shuffling the feature. That is, you may get different scores in different runs, which is confusing sometimes.

## Summary
* Error analysis helps us get insight into the erroneous cases of the model.
* Feature importance can be obtained via ablation studies regardless of the model type.


## Exercise
* Analyze the error distribution across different groups of the loan amount.
* Can you calculate the importance score of the batch norm for the model via ablation study?
