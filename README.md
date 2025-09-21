# Sentinal_Multimodals
SENTINEL Multimodals is an AI system that unifies images and text in a shared vector space using CLIP and Milvus. It enables fast cross-modal search, letting users query with text or images. With a Streamlit app and Docker support, it showcases scalable, real-world multimodal AI retrieval.
Architecture

This repository has four main modules:

Multimodal embeddings (green, orange): Core part of the engine. They allow us to represent multiple modalities (text, images) into a joint vector space. For this prototype, we are using the Jina CLIP v1 model
.

Vector store (blue): Based on Milvus
 vector database. This module stores all embeddings and powers the fast similarity search.

Retriever (purple): Handles the retrieval process by extracting embeddings and preparing a vector similarity search to retrieve the most relevant objects to the query. Works for both text and image queries.

App (brown): A Streamlit app that allows the user to interact with the search engine easily. The modules can also be re-used in more complex scenarios.

<p align="center"> <img src="assets/readme/multimodal-retrieval-architecture.png" align="middle" width="900" /> </p>
Core idea behind this project

This project focuses on using one single multimodal model (instead of chaining multiple models like image captioning + BERT). The Jina CLIP v1 model
 embeds both texts and images into the same space, enabling semantic comparison across modalities.

Multimodal models allow comparing a phrase like “a green apple” with an actual photo of a green apple and finding strong similarity — something impossible with older single-modality models.

<p align="center"> <img src="assets/readme/cross_modality.png" align="middle" width="400" /> </p>

SENTINEL Multimodal Retrieval is built on this concept: embed images in a multimodal space, and retrieve them using either text or image queries.

Set up

With Docker, setup is simple:

docker compose up -d


This pulls dependencies, builds the image, and launches the Streamlit app at http://localhost:8501/.

Since the vector database stores absolute paths, you’ll need to mount the volumes where your data is located using docker-compose.override.yaml:

services:
  sentinel-multimodal-retrieval:
    volumes:
      - "<absolute/path/to/your/data>:<absolute/path/to/your/data>"

Usage

Insert data into the vector database (e.g., Flickr8k or Flickr30k).

Run the app at http://localhost:8501/.

Search using either a text query or an image query.

Performance notes

Text-to-image search: ~9.5ms

Image-to-image search: ~14ms

Supports GPU acceleration for faster embeddings.

Roadmap for 2.x

Planned features:

Lighter dependencies (remove PyTorch/Transformers requirement).

Support for ONNX models.

Acknowledgments

Thanks to JinaAI
 for the CLIP model.

Thanks to Milvus
 for their vector database.
