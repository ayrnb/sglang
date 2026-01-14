from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='over_encoding_ops',
    packages=['over_encoding_ops'],
    ext_modules=[
        CUDAExtension(
            name='over_encoding_ops_kernel',
            sources=[
                'csrc/cpu_allocator.cpp', 
                'csrc/unified_memory_allocator.cpp', 
                'csrc/pybind_module.cpp',
            ],
            extra_compile_args={
                'cxx': ['-g'], 
                'nvcc': ['-O2', '--expt-relaxed-constexpr']
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
