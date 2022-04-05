# Clustering

## k-means

- Given a dataset $\{ x^{(1)}, \cdots, x^{(m)}\}$ where $x^{(i)} \in \mathbb{R}^n$
- Want to create $k$ clusters 
- [CS 229 Notes](https://cs229.stanford.edu/notes2021fall/cs229-notes7a.pdf)

### algorithm

1. randomly initialize cluster centroids $\mu_1, \cdots, \mu_k \in \mathbb{R}^n$
2. repeat until convergence
    1. for every $i$, set $c^{(i)} := argmin_j ||x^{(i)} - \mu_j||^2$
        - assign each datapoint to closest cluster centroid
    2. for every $j$ , $\mu_j = \frac{\sum 1(c^{(i)} = j) x^{(i)}}{\sum 1(c^{(i)} = j)}$
        - move each centroid to the avg position of the datapoints assigned to it 

### convergence

- guaranteed to converge
- Distortion Function: $J(c, \mu) = \sum ||x^{(i)} - \mu_{c^{(i)}}||^2$
    - Sum of the squared distances between the training samples and their assigned cluster centroid 
- k-means performs coordinate descent on $J$
- $J$ is nonconvex so might get trapped in local min
    - can run k-means multiple times with different intializations to escape local mins 



## hierarchical

- 