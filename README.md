# Image Embedding Association Test
A PyTorch implementation of ["Image Representations Learned With Unsupervised Pre-Training Contain Human-like Biases"](https://arxiv.org/pdf/2010.15052.pdf). In contrast to the [original implementation](https://github.com/ryansteed/ieat), I use [HuggingFace](https://huggingface.co/docs/transformers/model_doc/imagegpt)'s Transformers and support more vision models.

## Method

The iEAT adapts the [Word Embedding Association Test](https://www.science.org/doi/10.1126/science.aal4230) to the the image domain. Therefore, it computes the differential association using image-level embeddings where concepts and attributes are represented using a selected set of images.

## Results and Discussion 

I conducted the experiments using pre-trained BEiT-Base and ImageGPT-Large. 

|                 | Model         | Effect Size   | p-Value       |
| --------------- |:-------------:|:-------------:|:-------------:|
| Gender-Career   | BEiT-B        |      |       |  
| Gender-Career   | iGPT-L        |      |       |  
| Gender-Science  | BEiT-B        |      |       |  
| Gender-Science  | iGPT-L        |      |       |  
