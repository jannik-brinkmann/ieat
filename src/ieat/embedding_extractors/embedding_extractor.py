from typing import List

from PIL import Image
from transformers import AutoConfig, AutoFeatureExtractor


class PooledEmbeddingExtractor:

    def __init__(self, model_name_or_path):
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

    def __call__(self, images: List[Image.Image], *args, **kwargs):
        raise NotImplementedError
