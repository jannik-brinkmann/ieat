# Image Embedding Association Test (iEAT)
A PyTorch implementation of ["Image Representations Learned With Unsupervised Pre-Training Contain Human-like Biases"](https://arxiv.org/pdf/2010.15052.pdf). In contrast to the [original implementation](https://github.com/ryansteed/ieat), I use [HuggingFace](https://huggingface.co/docs/transformers/model_doc/imagegpt)'s ImageGPT API. 

The iEAT adapts the [Word Embedding Association Test](https://www.science.org/doi/10.1126/science.aal4230) (WEAT) to the image domain to determine the differential association of concepts X and Y with attributes A and B - s(X, Y, A, B). Therefore, it operates on pooled image-level embeddings where concepts and attributes are represented using a selected set of images. Thus, it makes the implicit assumption that categories can be represented using image sets.

Replication Results: 

|               | X             | Y             | A             | B             | Effect Size   | p-Value       |
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Age           | Young         | Old           | Pleasent      | Unpleasent    |               |               |  
