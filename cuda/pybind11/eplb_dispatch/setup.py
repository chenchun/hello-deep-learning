from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__version__ = "0.0.1"

setup(
    name="eplb_dispatch",
    ext_modules=[
        CUDAExtension(
            name="eplb_dispatch",
            sources=[
                "csrc/eplb/dispatch.cu",
                "csrc/eplb/dispatch_binding.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
