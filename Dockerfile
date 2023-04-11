# Dockerfile for Machine Learning Datasets: https://github.com/sheatsley/datasets
ARG BASE=pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM ${BASE}
COPY . mlds
RUN pip install --no-cache-dir ./mlds && rm -rf mlds
