# Dockerfile for Machine Learning Datasets: https://github.com/sheatsley/datasets
ARG BASE=pytorch/pytorch
FROM ${BASE}
COPY . mlds
RUN pip install --no-cache-dir ./mlds && rm -rf mlds
