"""Load one of the hybrid family's CUDA extensions (hybrid/kernels/*.cu).

The Modal image prebuilds each kernel into PREBUILT/<name>/ next to a copy of its
source; that build is used only when the copy matches this checkout's source
exactly. Otherwise the kernel is JIT-built, which needs nvcc matching torch.
"""

import functools
import importlib.util
import os
from pathlib import Path

KERNELS = Path(__file__).with_name("kernels")
PREBUILT = Path(os.environ.get("HYBRID_KERNELS_PREBUILT", "/opt/hybrid_kernels"))
FLAGS = ["-O3"]


def module_name(kernel):
    return f"hybrid_{kernel}"


@functools.cache
def load(kernel):
    """The compiled extension for hybrid/kernels/<kernel>.cu."""
    source = KERNELS / f"{kernel}.cu"
    built = PREBUILT / kernel
    copy = built / source.name
    if copy.exists() and copy.read_bytes() == source.read_bytes():
        path = built / f"{module_name(kernel)}.so"
        spec = importlib.util.spec_from_file_location(module_name(kernel), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from torch.utils.cpp_extension import load as build

    return build(name=module_name(kernel), sources=[str(source)], extra_cuda_cflags=FLAGS)
