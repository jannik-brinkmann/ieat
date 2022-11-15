from collections import OrderedDict


MODEL_MAPPING = OrderedDict(
    [
        ('beit', 'BeitEmbeddingExtractor'),
        ('imagegpt', 'ImageGPTEmbeddingExtractor'),
        ('vit', 'ViTEmbeddingExtractor'),
    ]
)


class AutoEmbeddingExtractor:

    _model_mapping = MODEL_MAPPING

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f'{self.__class__.__name__} is designed to be instantiated using .from_pretrained() method.'
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):



def _get_embedding_extractor(config, mapping):

