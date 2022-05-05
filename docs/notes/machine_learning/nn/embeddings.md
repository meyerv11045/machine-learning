# Embeddings

- Map items (e.g. movies, text, imgs, etc.) to low-dimensional real vectors such that similar items are close to each other

- Can be applied to dense data like audio to create a meaningful similarity metric

- Can jointly embedd multiple data types (e.g. text and images) to define a similarity between them

- embedding layer in a neural network is just a single hidden layer with one neuron for each dimension (usually $d$ dimensions)

- $ d \approx \text{possible vals}^ {1/4}$ 

    - rule of thumb, should be cross validated for particular datasets

    

## Continuous Retrieval

- Classical retrieval uses inverted index oftentimes on word
- Continuous retrieval represents objects as vectors in a vector space where similar objects are placed closer together 
- 3 Requirements:
    - 1. learn to represent objects as contiuous vectors (place everything in a learned space)
        2. learn to place similar objects close together
        3. learn to retrieve neighboring objects really fast (given a new query, encode it in the learned space and find its nearest neighbors)

## Resources

- [PyTorch Embedding Module Video](https://www.youtube.com/watch?v=euwN5DHfLEo)
- [Google Devs Embeddings Lecture](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)
- [Word Embeddings](https://ruder.io/word-embeddings-1/index.html)