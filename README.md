# Image Embedding Association Test
A PyTorch implementation of ["Image Representations Learned With Unsupervised Pre-Training Contain Human-like Biases"](https://arxiv.org/pdf/2010.15052.pdf). In contrast to the [original implementation](https://github.com/ryansteed/ieat), I use [HuggingFace](https://huggingface.co/docs/transformers/model_doc/imagegpt)'s Transformers and support additional vision models (ViT, BEiT). In addition, I integrate adapters trained using [AdapterHub](https://adapterhub.ml)'s Adapter-Transformers.

## Results and Discussion 

I conducted the experiments using pre-trained BEiT-Base and ImageGPT-Large. 

|                 | Model         | Effect Size   | p-Value       |
| --------------- |:-------------:|:-------------:|:-------------:|
| Gender-Career   | BEiT-B        |      |       |  
| Gender-Career   | iGPT-L        |      |       |  
| Gender-Science  | BEiT-B        |      |       |  
| Gender-Science  | iGPT-L        |      |       |  
