from abc import ABC, abstractmethod
from typing import Any, List

import numpy.typing as npt
from PIL import Image


class MultimodalFeatureExtractor(ABC):
    """Abstract base class for multimodal feature extractors.

    Defines the common interface for models that extract embeddings from images or text
    sentences, ensuring consistency in implementation and usage. Concrete feature extractor
    classes must inherit from this class and implement its methods.
    """

    @abstractmethod
    def get_image_features(self, images: List[Image.Image]) -> List[npt.NDArray]:
        """Extracts embeddings from the given image.

        Args:
            images (List[Image.Image]): Images to extract features from.

        Returns:
            List[npt.NDArray]: A list of numpy arrays containing the extracted embeddings

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def get_text_features(self, sentences: List[str]) -> List[npt.NDArray]:
        """Extracts embeddings from the given image.

        Args:
            sentences (List[str]): Sentences to extract features from.

        Returns:
            List[npt.NDArray]: A list of numpy arrays containing the extracted embeddings

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
