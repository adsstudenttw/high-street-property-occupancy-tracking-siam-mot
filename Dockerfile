FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    libopenblas-dev \
    libsm6 \
    libxext6 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv (https://docs.astral.sh/uv/)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

COPY requirements.txt requirements_exact.txt /tmp/
COPY docker/entrypoint.sh /usr/local/bin/siammot-entrypoint.sh

# Install Python + dependencies with uv.
# Python 3.8 is used for compatibility with this project's historical torch/apex stack.
RUN chmod +x /usr/local/bin/siammot-entrypoint.sh && \
    uv python install 3.8 && \
    uv venv --python 3.8 /opt/venv && \
    uv pip install --python /opt/venv/bin/python --upgrade pip setuptools wheel && \
    uv pip install --python /opt/venv/bin/python \
      --extra-index-url https://download.pytorch.org/whl/cu110 \
      torch==1.7.1+cu110 \
      torchvision==0.8.2+cu110 && \
    uv pip install --python /opt/venv/bin/python -r /tmp/requirements_exact.txt && \
    uv pip install --python /opt/venv/bin/python ninja && \
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install \
      --python /opt/venv/bin/python \
      --no-build-isolation \
      git+https://github.com/NVIDIA/apex.git

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

ENTRYPOINT ["/usr/local/bin/siammot-entrypoint.sh"]
CMD ["bash"]
