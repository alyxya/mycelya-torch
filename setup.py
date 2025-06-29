from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="torch_modal",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="torch_modal._C",
            sources=[
                "torch_modal/csrc/modal_extension.cpp",
            ],
            include_dirs=[],
            libraries=[],
            language="c++",
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
)