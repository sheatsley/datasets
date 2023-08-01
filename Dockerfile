# Dockerfile for Machine Learning Datasets: https://github.com/sheatsley/datasets
ARG BASE=nvidia/cuda:12.2.0-runtime-ubuntu22.04
FROM ${BASE}
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY . mlds
RUN pip3 install --no-cache-dir mlds/ && rm -rf mlds
