# Explainability of Linear Models

Linear models describe a target variable in terms of a linear combination of predictive variables. Models such as linear regression, lasso regression, logistic regression, and softmax regression are the popular variants of linear models :numref:`chap_linear`. The linearity assumption limits linear models' expressiveness but makes them easy-to-understand. Even though linear models are not perfect fit solutions oftentimes, they are extensively adopted in a multitude of domains especially when people are concerned more about what drives models' behaviors than predictive accuracy. Correctly interpreting linear models is a critical step to model understanding and is fundamental to more advanced explanation methods. 


## Revisit Linear Models 

To get started, let's revisit two popular linear models: linear regression and logistic regression. Linear regression is used to handle regression problems, whereas logistic regression is used to deal with classification problems.

Suppose we have $N$ samples and each sample has features $x_1^{(i)},...x_p^{(i)}$ and a groundtruth target $y^{(i)}$, $i=1,...,N$, the predicted output for this sample of linear regression is formulated as
$$
\hat{y}^{(i)} = b + w_1 x_1^{(i)} + ...+w_j x_j^{(i)}+...+ w_p x_p^{(i)},
$$

where $w_*$ represents the coefficient (or weight) that describe the mathematical relationship between each predictive variable and the predicted target; $b$ is the intercept which can also be viewed as a coefficient with $1$ as the feature value.

Logistic regression is useful for binary classification problems where the targets are dichotomic (e.g., absence or presence). It converts the output into probabilities using a logistic function.
$$
\hat{y}^{(i)} = \frac{1}{1 + \exp{(-(b + w_1 x_1^{(i)} +...+w_j x_j^{(i)}+...+ w_p x_p^{(i)}))}}.
$$
:eqlabel:`eq_logistic-regression`

Essentially, the parameters $w_*$ and $b$ of both models are learned by minimizing the discrepancies between predicted $\hat{y}^{(i)}$ and actual $y^{(i)}$ target values. For more details about the regularization, optimization, and evaluation process, please visit :numref:`chap_linear` and :numref:`chap_classification`. We import the necessary libraries below.

```{.python .input}
from d2l import torch as d2l
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
```

## Explaining Linear Models


### Interpreting Coefficients

Coefficients are the core of linear models, and they quantify the variation of the output when the given feature is varied, keeping all other features fixed. Interpreting the coefficients of linear regression models are straightforward. To begin, let's train a linear regression model, ridge regression which uses linear least squares as the loss and l2-norm as the regularization. We apply this model to predicting a patient's cholesterol level using the same heart disease dataset.

```{.python .input}
data = d2l.HeartDiseaseData(target='chol')
regr = linear_model.Ridge(solver='sag')
regr.fit(data.X_train, np.squeeze(data.y_train))
print(f'Mean Absolute Error: \
{mean_absolute_error(regr.predict(data.X_test),np.squeeze(data.y_test)):.5f}')
coef = {data.feat_col[i]: regr.coef_[i] for i in range(len(regr.coef_))}
print(f'Coefficients: {coef}; Intercept: {regr.intercept_}')
```

The coefficients tell us the conditional dependencies between a specific feature and the target when all other features remain constant. For instance, increasing feature $x_j$ by one unit will result in a change of $\hat{y}$ by $w_j$, when all other features remain unchanged. If $w_j$ is positive, $\hat{y}$ will be increased by $|w_j|$; otherwise, it will be decreased by this amount. If $x_j$ is a categorical variable, changing $x_j$ by one unit may represent switching it from one category to the other (e.g., from female=0 to male=1), but it also depends on how we encode the categories. If predictive variables are transformed before fitting the model, we need to know the precise detail of the transformation to concretely and precisely interpret the coefficients. In addition, the intercept $b$ anchors the regression line/surface in the right place and can be interpreted as the predicted output you would obtain when setting all features to zero. However, it may not have a meaningful interpretation since it is possible that $x_j$ cannot take the value 0 (e.g., blood pressure cannot be 0). 

In the following example, we increase a patient's age by one year and check the impact. It can be see that the  empirical analysis is identical to the theoretical interpretation.

```{.python .input}
index = data.feat_col.index('age')
patient = data.X_test[0]
patient_ageplus1 = patient.copy()
patient_ageplus1[index] = patient_ageplus1[index] + 1
print(f'Impact of increasing a patient\'s age: \
{regr.predict([patient_ageplus1])[0] - regr.predict([patient])[0]:.5f}.')
```

### Feature Importance
As you may notice, we used the raw data to fit the linear regression model, which means that the coefficients also took into account the natural units of the data. For example, *age* is expressed in "living years" while *resting blood pressure* is expressed in "mmHg". Using natural units is convenient for intuitive explanations, especially for tangible concepts such as weight, height, temperature, age, etc., but it is inappropriate to directly compare the coefficients to inspect the relative importance of input features. Furthermore, how you define the categorical features (e.g., different number of categories) will also impact the coefficient scale.

To make them comparable, we need to ensure the coefficients are unitless. We can achieve this by standardizing the raw data first and retraining the model to reduce all the coefficients to the same unit of measure. Or, we can multiply the coefficients by the standard deviation of the related features, which is equivalent to normalizing the input features,
$$
\hat{y} = b+\sum_{j}^{p} w_j \times x_j = b +\sum_{j}^{p} w_j (w_j \times \sigma_j) \times (\frac{x_j}{\sigma_j}).
$$

```{.python .input}
d2l.plot_hbar(data.feat_col, regr.coef_ * data.X_train.std(axis=0))
```

### Correlated Features

If one or multiple feature variables depend on or associate with another feature variable, we coin them as correlated features (also known as multicollinearity or collinearity). Correlated features will not bring additional information (or just very few). Instead, it might increase the model complexity and make it difficult to find a unique solution for linear models. Also, it will make the coefficient interpretation unreliable because the variables tend to change in unison. When features are correlated, changes in one variable are associated with variations in the other variable. The stronger the correlation, the more difficult it is to change one variable without changing another.

As such, it is good practice to check the correlation between variables in your dataset, and the correlation between two features can be measured by the Pearson, Kendall, or Spearman correlation coefficient :cite:`asuero2006correlation`. It is advised that we keep only one feature in the dataset if two features are highly correlated.

```{.python .input}
data.df['age'].corr(data.df['chol'], method='pearson')
```

Correlation can be negative, positive, or zero. A positive correlation signifies that both variables move in the same direction, while a negative correlation means that they move in the opposite direction. A correlation of zero indicates no relationship between the two variables being compared.

Some models are robust to correlated features. For instance, most tree-based models (e.g., decision trees and random forests) make no assumptions about the relationships between features. If two features are heavily correlated, one of them will be automatically ignored as splitting on it will have a low Gini or entropy value :cite:`Breiman.2001`. Some sophisticated post-hoc explanation methods such as SHAP :cite:`Lundberg.Lee.2017` can better deal with correlated features as it satisfies a vital property, *symmetry*: if two tokens contribute equally to all possible coalitions, their contribution values are the same.

### Odds Ratio
Interpreting logistic regression is more complicated than interpreting linear regression due to the nonlinear logistic function as the weights no longer influence the probability linearly. Unfortunately, we do not have a reasonable intuition about the logistic function, which thwarts the explanation the coefficients. Again, let's train a logistic regression model for heart disease prediction.

```{.python .input}
data = d2l.HeartDiseaseData()
lr = linear_model.LogisticRegression(solver='sag', max_iter=20000)
lr.fit(data.X_train, np.squeeze(data.y_train))
print(f'Validation Accuracy: \
    {lr.score(data.X_test, np.squeeze(data.y_test)):.5f}') 
```

To interpret this model, we will utilize odds and odds ratios. The odds of an event are the probability that the event occurs divided by the probability that the event does not occur. Odds are widely used in areas such as gambling and medical statistics.

In formal, let's reorganize the equation :eqref:`eq_logistic-regression` and move the linear combination part to the right.
$$
\ln (\frac{\mathbf{P}(y^{(i)}=1)}{1-\mathbf{P}(y^{(i)}=1) })= b + w_1 x_1^{(i)} +...+w_j x_j^{(i)}+...+ w_p x_p^{(i)}
$$

We can define odds as the predicted probability of having heart disease divided by the predicted probability of not having heart disease.
$$
\text{odds} = \frac{\mathbf{P}(y^{(i)}=1)}{1-\mathbf{P}(y^{(i)}=1) }=\frac{\mathbf{P}(y^{(i)}=1)}{\mathbf{P}(y^{(i)}=0) } = \exp (b + w_1 x_1^{(i)} +...+w_j x_j^{(i)}+...+ w_p x_p^{(i)})
$$

The odds ratio compares the odds of two events: increasing $x_j$ by one unit and keeping it unchanged, and is defined as
$$
\begin{aligned}
\frac{\text{odds} (x_j^{(i)}+1)}{\text{odds}} &= \frac{\exp (b + w_1 x_1^{(i)} + ...+w_j (x_j^{(i)}+1)+...+ w_p x_p^{(i)})}{\exp (b + w_1 x_1^{(i)} + ...+w_j x_j^{(i)}+...+ w_p x_p^{(i)})} \\
&= \exp (w_j (x_j^{(i)}+1) - w_j x_j^{(i)}) = \exp(w_j)
\end{aligned}
$$

Let's compute the odds of the logistic regression model.

```{.python .input}
pd.DataFrame(list(zip(data.feat_col, np.exp(lr.coef_[0]))), 
                  columns=['Feature', 'Odds Ratio']).transpose()
```

Now we can easily interpret logistic regression: increasing $x_j$ by one unit will scale the odds by a factor of $ \exp(w_j)$. For example, we observe that an increase in a patient's age does not change the odds of having heart disease vs. not having heart disease much since the odds ratio is around $1$. For categorical features such as chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic), changing the pain type from *typical angina* to  *atypical angina* will double the odds of having heart disease vs. not having heart disease because the odds ratio is over $2$. Note that the interpretations are merely about the model behavior and do not imply real-world causality.

## Summary
Linear models encompass a range of variants, such as ridge regression and logistic regression. Interpreting linear models is an essential yet non-trivial endeavor. We introduced how to correctly interpret two popular linear models (linear regression and logistic regression). When the underlying patterns are not linear, more complex models such as neural networks shall be used. In this case, linear models can still be utilized as surrogate models to explain the prediction of given instances :cite:`ribeiro2016should`.


## Exercises

1. When can we view coefficients as feature importance?
1. Can you interpret the feature *sex* in the logistic regression model?
1. Calculate the feature correlations bewteen features for the heart disease dataset.
