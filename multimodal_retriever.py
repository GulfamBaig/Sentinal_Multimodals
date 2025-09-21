import logging
from dataclasses import dataclass
from typing import List, Optional, Union

from PIL import Image
from pymilvus import MilvusClient

from feature_extraction import JinaCLIPFeatureExtractor, MultimodalFeatureExtractor
from vector_store.defaults import *

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """A dataclass representing a search hit.

    The caption is the ground truth caption obtained from the original dataset, the
    similarity is given by the vector database.
    """

    caption: str
    similarity: float
    filename: str


class MultimodalRetriever:
    def __init__(
        self,
        feature_extractor: Optional[MultimodalFeatureExtractor] = JinaCLIPFeatureExtractor(),
        db_uri: Optional[str] = MILVUS_URI,
        db_name: Optional[str] = MILVUS_DB_NAME,
        collection_name: Optional[str] = MILVUS_COLLECTION_NAME,
        top_k: Optional[int] = 20,
        output_fields: Optional[List[str]] = DEFAULT_FIELDS,
    ):
        """Initializes the multimodal retriever.

        Args:
            feature_extractor (Optional[MultimodalFeatureExtractor], optional): The feature extractor to use. Defaults to JinaCLIPFeatureExtractor().
            db_uri (Optional[str], optional): The URI of the Milvus server. Defaults to MILVUS_URI.
            db_name (Optional[str], optional): The name of the Milvus database. Defaults to MILVUS_DB_NAME.
            collection_name (Optional[str], optional): The name of the collection. Defaults to MILVUS_COLLECTION_NAME.
            top_k (Optional[int], optional): The number of results to return. Defaults to 20.
            output_fields (Optional[List[str]], optional): The fields to return in the output. Defaults to DEFAULT_FIELDS.
        """
        self.client = MilvusClient(uri=db_uri, db_name=db_name)
        self.feature_extractor = feature_extractor
        self.collection_name = collection_name
        self.top_k = top_k
        self.output_fields = output_fields

    def get_search_hits(
        self, search_query: Union[Image.Image, str], top_k: Optional[int] = None
    ) -> List[SearchHit]:
        """Searches the Milvus collection for the given query.

        Args:
            search_query (Union[Image.Image, str]): The query to search for. This query can be either an image or a string.
            top_k (Optional[int], optional): The number of results to return. Defaults to None.
        Returns:
            List[SearchHit]: A list of search hits, which include the caption, similarity and filename.
        """
        # extracting the embeddings depending on the type of the query
        if isinstance(search_query, str):
            query_embeddings = self.feature_extractor.get_text_features(search_query)
        elif isinstance(search_query, Image.Image):
            query_embeddings = self.feature_extractor.get_image_features(search_query)
        else:
            raise ValueError("The search query must be either a string or an Image object.")

        if top_k is None:
            top_k = self.top_k

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embeddings],
            output_fields=self.output_fields,
            limit=self.top_k,
            params={"metric_type": DEFAULT_METRIC},
        )
        hits: List[SearchHit] = []
        for hit in results[0]:
            # Since the DEFAULT_METRIC is COSINE, Milvus is already giving us similarity, not a distance
            hits.append(
                SearchHit(
                    caption=hit["entity"].get("caption"),
                    similarity=hit["distance"],
                    filename=hit["entity"].get("filename"),
                )
            )
        logger.info("Returning %s hits", len(hits))
        return hits

    def __del__(self):
        """Closes the Milvus client when the object is destroyed."""
        self.client.close()
