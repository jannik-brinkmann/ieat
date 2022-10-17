from PIL import Image

from collections import namedtuple
import os

from transformers import ImageGPTModel, ImageGPTFeatureExtractor
import torch
from typing import List





class ImageGPTPooledEmbeddingExtractor:

    def __init__(self, model_name):
        assert model_name in ('openai/imagegpt-' + s for s in ('small', 'medium', 'large')), 'Select a valid ImageGPT model.'

        self.model = ImageGPTModel.from_pretrained(model_name)
        self.feature_extractor = ImageGPTFeatureExtractor.from_pretrained(model_name)

    def __call__(self, images: List[Image.Image], n_layer, *args, **kwargs):

        features = self.feature_extractor(images, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**features, output_hidden_states=True, return_dict=True)

        # extract hidden state of the n-th layer
        hidden_state = output.hidden_states[n_layer]

        # average-pooling the layer norm across the sequence dimension
        return torch.mean(self.model.h[n_layer + 1].ln_1(hidden_state), 1)


class ImageEmbeddingAssociationTest:

    def __init__(self, embedding_extractor):
        self.embedding_extractor = embedding_extractor

    def __call__(self, specification: AssociationTestSpecification, *args, **kwargs):

        embeddings = self._extract_embeddings(specification)
        test = Test(*embeddings)
        test.run()

    def _extract_embeddings(self, specification: AssociationTestSpecification):
        embeddings = []

        # get directories of elements that are part of the association test
        dirs = [os.path.join('./data/ieat/', cat) for cat in (specification.X, specification.Y, specification.A, specification.B)]
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


if __name__ == "__main__":

    AssociationTestSpecification = namedtuple('AssociationTestSpecification', ['name', 'X', 'Y', 'A', 'B'])

    association_tests = (
        AssociationTestSpecification('Gender-Career', 'gender/male', 'gender/female', 'gender/career', 'gender/family')
    )

    embedding_extractor = ImageGPTPooledEmbeddingExtractor('openai/imagegpt-small')
    ieat = ImageEmbeddingAssociationTest(embedding_extractor)

    for a in association_tests:

        ieat(a)
