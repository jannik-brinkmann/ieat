import os
from PIL import Image

from .weat import WordEmbeddingAssociationTest


IMAGE_FILE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')


class ImageEmbeddingAssociationTest:

    def __init__(self, embedding_extractor):
        self.embedding_extractor = embedding_extractor

    def __call__(self, specification, *args, **kwargs):

        embeddings = self._extract_embeddings(specification)

        test = WordEmbeddingAssociationTest(*embeddings)

        test.run()

    def _extract_embeddings(self, specification):
        embeddings = []

        # get directories of elements that are part of the association test
        dirs = [os.path.join('./data/', cat) for cat in (specification.X, specification.Y, specification.A, specification.B)]
        for d in dirs:
            print("  " + str(d))

            # open all images that should represent the elements in the association test
            imgs = []
            image_paths = [os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1] in IMAGE_FILE_EXTENSIONS]
            for image_path in image_paths:

                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                imgs.append(img)

            # extract the average-pooled image embeddings from the model
            embeddings.append(self.embedding_extractor(imgs, 20))
        return embeddings