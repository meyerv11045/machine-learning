# Self Supervised Learning

- By forcing the model to learn a task defined in terms of input images, it forces the network to learn features which will be useful for downstream general visual recognition tasks later 

## Weakenss of Supervised Learning

- Annotations:
    - costly (e.g. medical imaging data)
    - ambiguous (set of labels not always obvious )
    - biased  (bias in selecting data and/or annotator)
    - privacy concerns (e.g. medical imaging data)
    - limits learning to predefined categories
        - e.g. all dogs are one category but later we might want to seperate dog breeds with transfer learning but this isnt great for that

## SSL Model Capacity

- Compared to SL, SSL benefits more from deeper models

    - also benefits from abundance of unlabeled data

- Can benefit from training a very deep model (teacher model) and then compressing it into a smaller model (student model)

    - this is done for categorical outputs from the teacher by training the students to mimic the output of the teacher network using something like KL divergence loss
    - when the teacher outputs non categorical outputs such as embeddings other loss functions need to be used such as MSE (performs not that well) or clustering with cross entropy 


## Pretext Tasks

- aka pseudo or proxy task
- Turn input to grayscale and make network colorize the image
    - learns features detectors for things such as grass and sky that are often the same color 



## Contrastive Learning

- Recent SSL technique
- Moco- Momentum Contrast Learning (CVPR 20)

1. From a query image it makes two augmented ones (e.g. two different crops)
2. The augmented ones are passed through two copies of the network (shared weights) to create two embeddings
3. The network is trained to pull together these augmented pairs (positive pairs)
4. In order to prevent the network from learning trivial embeddings, random images are passed through the network and the loss function is structured to push negative pairs away from the positive pairs 
    - A memory bank that remembers other negatives also helps
