from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='prc_custom_ops',
    packages=['prc_custom_ops'],
    find_packages=True,
    ext_modules=[
        CUDAExtension(
            name='prc_custom_ops_kernel',
            sources=[
                'csrc/cpu_allocator.cpp', 
                'csrc/overencoding_mtp.cu',
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
