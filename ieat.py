import os
from PIL import Image

from src import WordEmbeddingAssociationTest, ASSOCIATION_TESTS

import argparse


IMAGE_FILE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')


class ImageEmbeddingAssociationTest:

    def __init__(self, embedding_extractor):
        self.embedding_extractor = embedding_extractor

    def __call__(self, specification, *args, **kwargs):

        embeddings = self._extract_embeddings(specification)

        test = WordEmbeddingAssociationTest(*embeddings)
        e, p = test.run()

        return e, p

    def _extract_embeddings(self, s):
        embeddings = []

        directories = [os.path.join('./data/', c) for c in (s.X, s.Y, s.A, s.B)]
        for d in directories:

            images = []
            image_files = [os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1] in IMAGE_FILE_EXTENSIONS]
            for f in image_files:

                image = Image.open(f)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                images.append(image)

            embeddings.append(self.embedding_extractor(images))

        return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sum', dest='accumulate', action='store_const')
    args = parser.parse_args()

    embedding_extractor = AutoEmbeddingExtractor(args.model_name_or_path)
    ieat = ImageEmbeddingAssociationTest(embedding_extractor)

    for n, s in ASSOCIATION_TESTS.items():

        ieat(s)