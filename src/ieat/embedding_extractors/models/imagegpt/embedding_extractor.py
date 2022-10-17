from transformers import ImageGPTForCausalImageModeling

from ...embedding_extractor import PooledEmbeddingExtractor
from typing import List
from PIL import Image
import torch


class ImageGPTPooledEmbeddingExtractor(PooledEmbeddingExtractor):

    def __init__(self, model_name_or_path):
        super().__init__(model_name_or_path)

        self.imagegpt = ImageGPTForCausalImageModeling.from_pretrained(model_name_or_path)

    def __call__(self, images: List[Image.Image], *args, **kwargs):
        """
        ImageGPT authors extract image embeddings as
        1. use GPT-2 formulation to compute $ n^l = layer\_norm(h^l) $
        2. average pool $n^l$ across the sequence dimension: $ f^l = \langle n^l_i \rangle_i $

        see: https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf
        """
        features = self.feature_extractor(images, return_tensors="pt")

        l = kwargs.get('l', 20)
        with torch.no_grad():

            # extract hidden states of images
            outputs = self.imagegpt(**features, output_hidden_states=True, return_dict=True)

            # extract l-th hidden state $h^l$
            h_l = outputs.hidden_states[l]

            # use GPT-2 formulation to compute $ n^l = layer\_norm(h^l) $
            n_l = self.imagegpt.transformer.h[l + 1].ln_1(h_l)

            # average pool $n^l$ across the sequence dimension
            embeddings = torch.mean(n_l, 1)

        return embeddings
