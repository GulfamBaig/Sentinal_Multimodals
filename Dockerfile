FROM nvidia/cuda:12.6.1-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libsm6 \
    libxext6 \
    ## Python
    python3-dev \
    python3-numpy \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY . ./sentinel-workspace
WORKDIR /sentinel-workspace

RUN pip install --upgrade pip

RUN pip3 install -r requirements.txt

RUN python3 utils/download_models.py "jinaai/jina-clip-v1"