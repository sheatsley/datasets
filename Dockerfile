# Dockerfile for Machine Learning Datasets: https://github.com/sheatsley/datasets
FROM tensorflow/tensorflow
FROM pytorch/pytorch
COPY . /mlds
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN cd /mlds && pip install --no-cache-dir -e .
