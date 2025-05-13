from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="dispatch_cuda",
    ext_modules=[
        CUDAExtension(
            name="dispatch_cuda",
            sources=["dispatch_cuda.cpp"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)