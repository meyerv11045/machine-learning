# Feature Scaling

- Datasets can have features with varying magnitudes, ranges, and units
- Pre-processing the features can improve model performance
    - decision tree based algorithms are invariant to feature scaling due to making decisions at each node based on a single feature

## Use when

- optimizing using gradient desecent (e.g. regressions, NNs)
    - feature value in the cost function derivative and gradient step will cause different step sizes for each feature due to the difference in feature ranges
    - scale the data to ensure params are updated at the same rate for all features
    - similarly scaled features can make it converge faster
- applying distance based algorithms (e.g. KNN, SVMs)
    - using distance btw data points as a measure of similarity will bias the measure towards higher magnitude features
    - scaling data ensures unbiased measure of similarity


## Normalization

- Values are shifted and rescaled to the range $[0,1]$
- aka min-max scaling
- $$X' = \frac{X - X_{min}}{X_{max} -X_{min}}$$
- where $X_{min}$ and $X_{max}$ are the min and max values of the feature
    - more affected by outliers

```python
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler().fit(x_train)

std_x_train = scale.transform(x_train)
std_x_test = scale.transform(x_test)  

#can also compute the fit and then transform in one step
std_x_train = MinMaxScaler().fit_transform(x_train)
```

### Use when

- data is not gaussianly distributed
    - useful in algorithms that don't assume any distribution of the data (KNN, neural nets)

## Standardization

- Values are centered around the mean with a unit standard deviation
    - new mean of feature is 0 and new standard deviation is 1
- $$X' = \frac{X - \mu}{\sigma}$$
- where $\mu$ is the mean of that feature's values and $\sigma$ is the standard deviation
- values are not restricted to a particular range
    - more robust to outliers

``` python
from sklearn.preprocessing import StandardScaler

scale = StandardScaler().fit(x_train)

std_x_train = scale.transform(x_train)
std_x_test = scale.transform(x_test)  
```

### Use when

- data is gaussianly distributed
    - can be useful in some cases where this isn't true

## Notes

- Normalization vs. Standardization depends on the problem and the learning algorithm being used
- Best practice to scale the training data and use the same values to scale the test set
- Don't need to scale the target values normally



## Resources

- [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)
- [Machine Learning Mastery](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)
- [Scikit-Learn Preprocessing Docs](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)