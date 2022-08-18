# Image Embedding Association Test (iEAT)
A PyTorch implementation of ["Image Representations Learned With Unsupervised Pre-Training Contain Human-like Biases"](https://arxiv.org/pdf/2010.15052.pdf). In contrast to the [original implementation](https://github.com/ryansteed/ieat), I use [HuggingFace](https://huggingface.co/docs/transformers/model_doc/imagegpt)'s ImageGPT API. 

## Method

The iEAT adapts the [Word Embedding Association Test](https://www.science.org/doi/10.1126/science.aal4230) to the image domain to determine the differential association of concepts X and Y with attributes A and B - s(X, Y, A, B). Therefore, it operates on pooled image-level embeddings where concepts and attributes are represented using a selected set of images. Thus, it makes the implicit assumption that categories can be represented using image sets.

## Results and Discussion 

|                 | X                 | Y                | A             | B             | Effect Size   | p-Value       |
| --------------- |:-----------------:|:----------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Age             | Young             | Old              | &nbsp;&nbsp;&nbsp;Pleasent&nbsp;&nbsp;&nbsp;      | Unpleasent    | -0.372293     | 0.737013      |  
| Arab-Muslim     | Other             | Arab-Muslim      | Pleasent      | Unpleasent    | 1.00325       | 0.011         |  
| Asian           | European American | Asian American   | American      | Foreign       | 0.414446      | 0.244589      |  
| Disability      | Disabled          | Abled            | Pleasent      | Unpleasent    | -0.372293     | 0.737013      |  
| Gender-Career   | Male              | Female           | Career        | Family        | -0.372293     | 0.737013      |  
| Gender-Science  | Male              | Female           | Science       | Liberal Arts  | -0.372293     | 0.737013      |  
| Insect-Flower   | Flower            | Insect           | Pleasent      | Unpleasent    | -0.372293     | 0.737013      |  
| Native          | European American | Native American  | U.S.          | World         | -0.372293     | 0.737013      |  
| Race            | European American | African American | Pleasent      | Unpleasent    | -0.372293     | 0.737013      |
| Religion        | Christianity      | Judaism          | Pleasent      | Unpleasent    | -0.372293     | 0.737013      |
| Sexuality       | Gay               | Straight         | Pleasent      | Unpleasent    | -0.372293     | 0.737013      |
| Skin-Tone       | Light             | Dark             | Pleasent      | Unpleasent    | -0.372293     | 0.737013      |
| Weapon          | White             | Black            | Tool          | Weapon        | -0.372293     | 0.737013      |
| Weapon (Modern) | White             | Black            | Tool          | Weapon        | -0.372293     | 0.737013      |
| Weight          | Thin              | Fat              | Pleasent      | Unpleasent    | -0.372293     | 0.737013      |
