FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="8.6"

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

# Install uv (https://docs.astral.sh/uv/).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
COPY docker/entrypoint.sh /usr/local/bin/siammot-entrypoint.sh

# Follow SiamMOT INSTALL.md ordering: torch/torchvision first, then requirements.txt.
# Match the container CUDA toolkit to PyTorch cu110 so Apex can build its CUDA extensions.
# Build maskrcnn-benchmark with FORCE_CUDA=1 because docker build does not expose the host GPU,
# and its setup.py otherwise falls back to CPU-only custom ops. Also set TORCH_CUDA_ARCH_LIST so
# torch does not try to query a live GPU for architecture detection during docker build.
# Apex is pinned to a torch==1.7.1+cu110-compatible revision instead of current HEAD.
# Patch a couple of newer Apex Torch API checks so `import apex` still works on Torch 1.7.1.
RUN chmod +x /usr/local/bin/siammot-entrypoint.sh && \
  uv python install 3.8 && \
  uv venv --python 3.8 /opt/venv && \
  uv pip install --python /opt/venv/bin/python --upgrade pip setuptools wheel && \
  uv pip install --python /opt/venv/bin/python \
  -f https://download.pytorch.org/whl/torch_stable.html \
  torch==1.7.1+cu110 \
  torchvision==0.8.2+cu110 && \
  uv pip install --python /opt/venv/bin/python \
  numpy==1.22.0 \
  Pillow==9.5.0 \
  cython==3.0.11 && \
  grep -v '^pycocotools' /tmp/requirements.txt > /tmp/requirements.docker.txt && \
  uv pip install --python /opt/venv/bin/python \
  --no-build-isolation \
  -r /tmp/requirements.docker.txt && \
  uv pip install --python /opt/venv/bin/python ninja matplotlib opencv-python cityscapesscripts && \
  git clone --depth 1 https://github.com/cocodataset/cocoapi.git /opt/src/cocoapi && \
  bash -lc "cd /opt/src/cocoapi/PythonAPI && /opt/venv/bin/python setup.py build_ext install" && \
  git clone --depth 1 https://github.com/facebookresearch/maskrcnn-benchmark.git /opt/src/maskrcnn-benchmark && \
  cuda_dir="/opt/src/maskrcnn-benchmark/maskrcnn_benchmark/csrc/cuda" && \
  perl -i -pe 's/AT_CHECK/TORCH_CHECK/' "$cuda_dir/deform_pool_cuda.cu" "$cuda_dir/deform_conv_cuda.cu" && \
  bash -lc "cd /opt/src/maskrcnn-benchmark && PATH=/opt/venv/bin:\$PATH FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} /opt/venv/bin/python setup.py build develop" && \
  PATH="/opt/venv/bin:${PATH}" APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install --python /opt/venv/bin/python \
  --no-build-isolation \
  git+https://github.com/NVIDIA/apex.git@da9f5ae && \
  apex_norm_file="/opt/venv/lib/python3.8/site-packages/apex/normalization/fused_layer_norm.py" && \
  perl -0pi -e 's/return hasattr\(torch\.library, "custom_op"\)/return hasattr(getattr(torch, "library", None), "custom_op")/' "$apex_norm_file" && \
  perl -0pi -e 's/torch\.compiler\.is_compiling\(\)/(hasattr(torch, "compiler") and torch.compiler.is_compiling())/g' "$apex_norm_file"

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/workspace"

ENTRYPOINT ["/usr/local/bin/siammot-entrypoint.sh"]
CMD ["bash"]
