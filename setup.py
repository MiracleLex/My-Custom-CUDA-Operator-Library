# setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='MyCustomCUDAOperatorLibrary',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='MyCustomCUDAOperatorLibrary._C',
            sources=[
                'csrc/bindings.cpp',
                'csrc/ops/elementwise.cu',
                'csrc/ops/reduce.cu',
                'csrc/ops/sgemm.cu',
                'csrc/ops/flash_attention.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch>=1.9.0'],
)