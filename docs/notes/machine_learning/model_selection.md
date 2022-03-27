# Model Selection

- Learning algorithms have many different hyperparameters that can be tuned
- We want to select the hyperparameters that lead to the best models
- We utilize algorithms to perform an optimizing search in the space of possible models to determine the bset one
- Given a set of models $M = \{M_1, \dots \}$

## Cross Validation

### Hold-Out (Simple)

- Hold-out cross validation (aka simple cross validation):

	1. Randomly slit the train set $S$ into a $S_{train}$ and $S_{cv}$ (the hold-out cross validation set)
	1. Train each model $M_i$ on $S_{train}$ to get a hypothesis $h_i$
	1. Select $h_i$ with the smallest error on the hold out cross validation set $S_{cv}$

- Can optionally retrain the best model $h_i$ on the entire train set $S$
    - Even then, we are still selecting best model based on $0.8m$ training examples rather than $m$
- Wastes some of the dataset on holding out- bad for situations with scare data (e.g. $m =20$)

### k-fold 

- Holds out less data compared to simple cv

1. Randomly split train set $S$ into $k$ disjoint subsets $S_1, \dots, S_k$ of $m / k$ training examples each
2. For each model $M_i$ evaluate:
    - for each of the $k$ folds, hold out $S_j$ and train the model on all the other folds
    - test the hypothesis $h_{ij}$ on $S_j$ to get the error
    - the estimated generalization error of model $M_i$ is the average of the errors over each if the $k$ tests
3. Pick the model $M_i$ with the lowest estimated generalization error and retrain the model on the entire training set $S$. The resulting hypothesis is our final output

- Typical choice is $k =10$
- Computationally more expensive than hold-out since we need to train *each* model $k$ times 

### Leave-one-out

- Useful for when data is scare
- Perform k-fold cross validation where $k = m$ in order to leave out as little data as possible each time
- Holds out one training example each time and then averages together the resulting $m = k$ errors to get an estimate for the generalization error of a model

### Nested Cross Validation 

- Used to do model hyperparameter optimization and model selection at the same time while avoiding overfitting the training data
- k-fold CV for model hyperparam optimization is nested inside th k-fold CV for model selection
    - this prevents hyperparam search from overfitting the dataset since it is exposed to only a subset of the data provided by the outer CV procedure
- Very computationally expensive



## Feature Selection

- If $n >> m$, its best to reduce the number of features to learn from
- Reducing the number of input variables to produce simpler models
    - simpler models = more explainable since decisions are made based on less features
- Reduces overfitting
- Reduces training time/compute time
- For $n$ features, there are $2^n$ possible feature subsets
    - Each of the $n$ features can be included or excluded from the subset 
- Feature selection can be viewed as a model selection problem over $2^n$ possible models
- For large $n$, its too expensive to compare all $2^n$ models explicitly 
    - Therefore, a heuristic search procedure is used to find a good feature subset

### Wrapper

- Procedure that wraps your learning algorithm
- Dependent on the algorithm you are selecting features for
    - Different learning algorithms may have different best features from the same wrapper algorithm
- More computationally expensive $O(n^2)$ calls to learning algorithm

#### Forward Search

- Makes assumption that best answer is a small set of features, likely those that come first
    - Order the features are specified in matters
- Unable to remove feature
- Define feature set  $F = \empty$ to be the indices of the features that matter
- Repeat until $|F| = n$ or $|F| >$ max features:
    - For each feature $i = 1, \dots, n$:
        - train a model using $F \cup \{i\}$
        - measure the generalization using some version of cross validation
    - Set $F$ to be the best feature subset found in the previous step

#### Backward Search

- Makes assumption that best answer is a near full set of the features
- Spends most time checking large subsets
- Cannot reinstate removed features
- Define feature set  $F = \{1, \dots, n \}$
- Repeat until $|F| = \empty$:
    - For each feature $i = 1, \dots, n$:
        - train a model using $F \setminus \{i\}$
        - measure the generalization using some version of cross validation
    - Set $F$ to be the best feature subset found in the previous step

### Filter

- Computationally cheaper than wrapper method
- Compute a score on the data that measures the effectiveness 
- Indepedent of the underyling learning algorithm in determining most important features
- Define some score $S(j)$ which measure how informative feature $j$ is for the output 
- Choose the features with the largest scores
- This heuristic for selecting intital features can be used to select an initial set of features to use with forward and backward search
- Example heuristics:
    - mutual infromation
    - absolute value of the correletion

#### Mutual Information

- Common scoring heuristic (especially for discrete-valued features)
- Can be expressed as Kullback-Leibler (KL) divergence which measures the difference between two probability distributions



## Resources

- [Feature Selection (ML Mastery)](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)