# Image Embedding Association Test (iEAT)
A PyTorch replication of ["Image Representations Learned With Unsupervised Pre-Training Contain Human-like Biases"](https://arxiv.org/pdf/2010.15052.pdf). In contrast to the [original implementation](https://github.com/ryansteed/ieat), I use [HuggingFace](https://huggingface.co/docs/transformers/model_doc/imagegpt)'s ImageGPT API. 

## Method

The iEAT adapts the [Word Embedding Association Test](https://www.science.org/doi/10.1126/science.aal4230) to the the image domain. Therefore, it computes the differential association of concepts X and Y with attributes A and B using pooled image-level embeddings where concepts and attributes are represented using a selected set of images. In consequence, it is based on the implicit assumption that categories can be represented using image sets.

## Results and Discussion 

I conducted the experiments using pre-trained ImageGPT-Large. The obtained differential associations often differ from the results reported in the original paper. The source of these deviations is still under investigation. 

|                 | X                 | Y                | A             | B             | Effect Size   | p-Value       |
| --------------- |:-----------------:|:----------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Age             | Young             | Old              | Pleasent      | Unpleasent    | -0.37     | 0.74      |  
| Arab-Muslim     | Other             | Arab-Muslim      | Pleasent      | Unpleasent    | 1.00          | 0.01          |  
| Asian           | European American | Asian American   | &nbsp;&nbsp;American&nbsp;&nbsp;      | Foreign       | 0.42          | 0.25          |  
| Disability      | Disabled          | Abled            | Pleasent      | Unpleasent    | 0.84          | 0.14          |  
| Gender-Career   | Male              | Female           | Career        | Family        |      |       |  
| Gender-Science  | Male              | Female           | Science       | Liberal Arts  |      |       |  
| Insect-Flower   | Flower            | Insect           | Pleasent      | Unpleasent    |      |       |  
| Native          | European American | &nbsp;&nbsp;Native American&nbsp;&nbsp;  | U.S.          | World         |      |       |  
| Race            | European American | African American | Pleasent      | Unpleasent    |      |       |
| Religion        | Christianity      | Judaism          | Pleasent      | Unpleasent    |      |       |
| Sexuality       | Gay               | Straight         | Pleasent      | Unpleasent    |      |       |
| Skin-Tone       | Light             | Dark             | Pleasent      | Unpleasent    |      |       |
| Weapon          | White             | Black            | Tool          | Weapon        |      |       |
| Weapon (Modern) | White             | Black            | Tool          | Weapon        |      |       |
| Weight          | Thin              | Fat              | Pleasent      | Unpleasent    |      |       |

