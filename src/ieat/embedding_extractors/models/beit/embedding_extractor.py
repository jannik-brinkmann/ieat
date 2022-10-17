from transformers import BeitForMaskedImageModeling

from ...embedding_extractor import PooledEmbeddingExtractor
from typing import List
from PIL import Image
import torch


class BeitPooledEmbeddingExtractor(PooledEmbeddingExtractor):

    def __init__(self, model_name_or_path):
        super().__init__(model_name_or_path)

        self.beit = BeitForMaskedImageModeling.from_pretrained(model_name_or_path)

    def __call__(self, images: List[Image.Image], *args, **kwargs):
        """
        BEiT authors follow iGPT authors and extract image embeddings as
        1. compute $ n^l = layer\_norm(h^l) $ (excluding [CLS] token)
        2. average pool $n^l$ across the sequence dimension: $ f^l = \langle n^l_i \rangle_i $

        see: https://arxiv.org/pdf/2106.08254.pdf
        """
        features = self.feature_extractor(images, return_tensors="pt")

        l = kwargs.get('l', 9)  # BEiT-Base; l = kwargs.get('l', 14) for BEiT-Large
        with torch.no_grad():

            # extract hidden states of images
            outputs = self.beit(**features, output_hidden_states=True, return_dict=True)

            # extract l-th hidden state $h^l$ (excluding [CLS] token)
            h_l = outputs.hidden_states[l][:, 1:, :]

            # compute $ n^l = layer\_norm(h^l) $
            n_l = self.beit.beit.encoder.layer[l + 1].layernorm_before(h_l)

            # average pool $n^l$ across the sequence dimension
            embeddings = torch.mean(n_l, 1)

        return embeddings
