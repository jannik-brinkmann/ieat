from PIL import Image
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
from typing import List


class EmbeddingExtractor:

    def __init__(self, model_name_or_path):
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_config(config)

    def __call__(self, images: List[Image.Image], *args, **kwargs):
        raise NotImplementedError
