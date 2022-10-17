from src.ieat import ImageEmbeddingAssociationTest, ASSOCIATION_TESTS, ImageGPTPooledEmbeddingExtractor, BeitPooledEmbeddingExtractor, ViTPooledEmbeddingExtractor


if __name__ == "__main__":

    # embedding_extractor = ImageGPTPooledEmbeddingExtractor('openai/imagegpt-large')
    # embedding_extractor = BeitPooledEmbeddingExtractor('microsoft/beit-base-patch16-224-pt22k')
    embedding_extractor = ViTPooledEmbeddingExtractor('facebook/dino-vitb16')
    ieat = ImageEmbeddingAssociationTest(embedding_extractor)

    for name, specification in ASSOCIATION_TESTS.items():

        ieat(specification)