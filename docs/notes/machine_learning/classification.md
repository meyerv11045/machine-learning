# Classification

## Logistic Regression

## Softmax/Multinomial Logistic Regression

- [Stanford UFLDL](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/)
- Learn after exam

## Gaussian Discriminant Analysis

## Naive Bayes

- Features are discrete unlike the continue features in Gaussian Discriminant Analysis

- Ex: dictionary of words showing up in an email to classify whether it is spam 

    - mispelled words in spam messages to try to get past these dictionaries of possible spam values

    - the feature vector of words that show up in the desired dictionary is considered a bernoulli event model 

    - a multinomial event model takes into account the structure of the sentence or how the words appear in order to help prevent stuffing an email with a bunch of hidden good words that help the email get past the spam filter using a bernoulli event model
        - naive bayes is no longer valid
            - the feature vector will now depend on email length
            - laplace mothing based on dictionary size instead of the size of the feature vector

## Support Vector Machines

## Multiclass Classification

- To classify an input into one of $k$ classes, there are two techniques that can be applied to any of the classification methods (logistic regression, perceptron, SVM)
- [ML Mastery](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

### One-vs-All

- aka one-vs-rest
- For each $i = 1, \dots, k$ train a binary classifier to succesfully classify $y = i$. 
    - This yields a set of parameters $\theta^{(1)}, \dots, \theta^{(k)}$

- To make predictions for $x$, select the class $i$ that produces the highest value for $h_\theta^{(i)}(x) = (\theta^{(i)})^Tx$

### One-vs-One

- For each class $i = 1, \dots, k$ train a separate binary classifier to succesfully classify $y = i$ against each other class
    - Results in $\frac{k(k-1)}{2}$ Classifiers 
-  To make predictions for $x$:
    - Each model may predict a class label for $x$ and the class label with the most votes (most frequently occuring) amongst all the models is the predicted label
    - Each model may predict a probability of a class membership of $x$, so then the probabilites for each class membership are summed up and the class with the highest probability is the predicted label.
- Example w/ 4 classes: ‘*red*,’ ‘*blue*,’ and ‘*green*,’ ‘*yellow*.’ This would be divided into six binary classification problems:
    - 1: red vs. blue
    - 2: red vs. green
    - 3: red vs. yellow
    - 4: blue vs. green
    - 5: blue vs. yellow
    - 6: green vs. yellow