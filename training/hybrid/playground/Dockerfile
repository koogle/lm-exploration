# GPU Space for the hybrid playground: CUDA 13.0 devel (nvcc for the family's kernels),
# torch 2.14.1+cu130, the kernels prebuilt where hybrid/cuda_extension.py looks.
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HYBRID_KERNELS_PREBUILT=/opt/hybrid_kernels
RUN uv venv --python 3.12 /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt

COPY model_hybrid/kernels /tmp/kernels
RUN for kernel in softmax_gradient kda_convolution decode; do \
      mkdir -p /opt/hybrid_kernels/$kernel && cp /tmp/kernels/$kernel.cu /opt/hybrid_kernels/$kernel/ && \
      python -c "from torch.utils.cpp_extension import load; load(name='hybrid_$kernel', sources=['/opt/hybrid_kernels/$kernel/$kernel.cu'], build_directory='/opt/hybrid_kernels/$kernel', extra_cuda_cflags=['-O3'])"; \
    done

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
COPY --chown=user . .
CMD ["python", "app.py"]
