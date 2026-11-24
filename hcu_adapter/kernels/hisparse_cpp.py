# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Build specification for the optional precompiled HIP HiSparse provider.

Only the explicit native build entry point calls extension(); importing this
module neither imports Torch nor starts a compiler or device runtime.
"""


def extension(root, *, libraries, extra_link_args):
    from torch.utils.cpp_extension import CUDAExtension

    return CUDAExtension(
        name='sgl_kernel.hisparse_ops',
        sources=[str(root / 'csrc/hisparse/hisparse_cpp.cu')],
        libraries=libraries,
        extra_link_args=extra_link_args,
        extra_compile_args={'cxx': ['-O2', '-std=c++17'],
                            'nvcc': ['-O2', '-std=c++17', '-DUSE_ROCM=1']},
        py_limited_api=False,
    )
