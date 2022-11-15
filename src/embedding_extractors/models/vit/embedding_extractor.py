from typing import List

import torch
from PIL import Image

from ...embedding_extractor import EmbeddingExtractor


class ViTEmbeddingExtractor(EmbeddingExtractor):

    def __init__(self, model_name_or_path):
        super().__init__(model_name_or_path)

    def __call__(self, images: List[Image.Image], *args, **kwargs):
        features = self.feature_extractor(images, return_tensors="pt")

        l = kwargs.get('l', 12)
        with torch.no_grad():
            outputs = self.model(**features, output_hidden_states=True, return_dict=True)

            # extract l-th hidden state $h^l$ (excluding [CLS] token)
            h_l = outputs.hidden_states[l][:, 1:, :]

            # use GPT-2 formulation to compute $ n^l = layer\_norm(h^l) $
            n_l = self.model.transformer.h[l + 1].ln_1(h_l)

            # average pool $n^l$ across the sequence dimension
            embeddings = torch.mean(n_l, 1)

        return embeddings
