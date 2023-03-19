# Dockerfile for Machine Learning Datasets: https://github.com/sheatsley/datasets
FROM tensorflow/tensorflow
FROM pytorch/pytorch
COPY . /mlds
RUN cd /mlds && pip install -e .
