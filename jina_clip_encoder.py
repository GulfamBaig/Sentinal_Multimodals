from typing import List, Optional

import numpy.typing as npt
import torch
from PIL import Image
from transformers import AutoModel

from .feature_extractor import MultimodalFeatureExtractor


class JinaCLIPFeatureExtractor(MultimodalFeatureExtractor):
    def __init__(self, model_name: Optional[str] = "jinaai/jina-clip-v1") -> None:
        """Initializes the Jina CLIP feature extractor. The embeddings returned from this model
        were optimized for cosine similarity.

        Full docs here: https://huggingface.co/jinaai/jina-clip-v1

        Args:
            model_name (Optional[str], optional): The model name to use. Defaults to "jinaai/jina-clip-v1".
        """
        self._init_model(model_name)

    def _init_model(self, model_name: str) -> None:
        """Initializes the huggingface model into self.model. The model is loaded on the GPU if
        available, CPU otherwise.

        Args:
            model_name (str): The model name to use.

        Raises:
            Exception: If the model could not be loaded.
        """
        # check if a GPU is available, if not, torch will use the CPU
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # initialize the huggingface model
        try:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        except Exception as e:
            raise Exception(f"Could not load model {model_name}. Error: {e}")

    def get_image_features(self, images: List[Image.Image]) -> List[npt.NDArray]:
        """Extracts embeddings from a list of of images.

        Args:
            images (List[Image.Image]): Images to extract features from.

        Returns:
            List[npt.NDArray]: A list of numpy arrays containing the extracted embeddings
        """
        if isinstance(images, Image.Image):
            images = [images]

        # get image embeddings, remove batch dim
        embeddings = self.model.encode_image(images).squeeze()

        return embeddings

    def get_text_features(self, sentences: List[str]) -> List[npt.NDArray]:
        """Extracts embeddings from a list of strings.

        Args:
            sentences (List[str]): Sentences to extract features from.

        Returns:
            List[npt.NDArray]: A list of numpy arrays containing the extracted embeddings
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        # get text embeddings, remove batch dim
        embeddings = self.model.encode_text(sentences).squeeze()

        return embeddings
