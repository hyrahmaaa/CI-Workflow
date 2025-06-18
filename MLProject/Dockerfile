FROM continuumio/miniconda3:latest

WORKDIR /app

COPY conda.yaml .

RUN conda env create -f conda.yaml

ENV PATH="/opt/conda/envs/mlproject-env/bin:$PATH"

COPY . .