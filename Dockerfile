FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda
ARG CONDA_ENV=siammot

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
  perl \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
  bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
  rm -f /tmp/miniconda.sh && \
  "${CONDA_DIR}/bin/conda" config --system --set auto_update_conda false && \
  "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
  "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
  "${CONDA_DIR}/bin/conda" clean -afy

ENV PATH="${CONDA_DIR}/bin:${PATH}"

WORKDIR /workspace

COPY requirements.txt requirements_exact.txt /tmp/
COPY docker/entrypoint.sh /usr/local/bin/siammot-entrypoint.sh

# Python 3.8 is used for compatibility with this project's historical torch/apex stack.
RUN chmod +x /usr/local/bin/siammot-entrypoint.sh && \
  conda create -y -n "${CONDA_ENV}" python=3.8 pip && \
  conda run -n "${CONDA_ENV}" python -m pip install --upgrade pip setuptools wheel && \
  conda run -n "${CONDA_ENV}" pip install \
  -f https://download.pytorch.org/whl/torch_stable.html \
  torch==1.7.1+cu110 \
  torchvision==0.8.2+cu110
#   conda run -n "${CONDA_ENV}" pip install -r /tmp/requirements_exact.txt && \
#   conda run -n "${CONDA_ENV}" pip install cython ninja
# RUN git clone --depth 1 https://github.com/facebookresearch/maskrcnn-benchmark.git /opt/src/maskrcnn-benchmark && \
#   cuda_dir="/opt/src/maskrcnn-benchmark/maskrcnn_benchmark/csrc/cuda" && \
#   perl -i -pe 's/AT_CHECK/TORCH_CHECK/' "$cuda_dir/deform_pool_cuda.cu" "$cuda_dir/deform_conv_cuda.cu" && \
#   conda run -n "${CONDA_ENV}" bash -lc "cd /opt/src/maskrcnn-benchmark && python setup.py build develop" && \
#   APEX_CPP_EXT=1 APEX_CUDA_EXT=1 conda run -n "${CONDA_ENV}" pip install \
#   --no-build-isolation \
#   git+https://github.com/NVIDIA/apex.git && \
#   conda clean -afy

# ENV CONDA_ENV="${CONDA_ENV}"
# ENV PATH="${CONDA_DIR}/envs/${CONDA_ENV}/bin:${CONDA_DIR}/bin:${PATH}"
# ENV PYTHONPATH="/workspace"

# ENTRYPOINT ["/usr/local/bin/siammot-entrypoint.sh"]
# CMD ["bash"]
