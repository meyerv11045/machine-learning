# Classification

## Logistic Regression

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