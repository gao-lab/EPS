FROM nvcr.io/nvidia/rapidsai/base:25.04-cuda12.8-py3.12

# Update and install dependencies
USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install gcc g++ git -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy package and install
USER rapids
WORKDIR /app
COPY ./expression_copilot ./expression_copilot
COPY ./README.md ./README.md
COPY ./pyproject.toml ./pyproject.toml
RUN pip --no-cache-dir install -e "." && rm -rf /tmp/*
