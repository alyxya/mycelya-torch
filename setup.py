# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import distutils.command.clean
import os
import platform
import shutil
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext


PACKAGE_NAME = "torch_remote"
version = "0.1.0"

ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / "torch_remote/csrc"


def get_build_ext_class():
    """Get PyTorch's BuildExtension class, importing only when needed."""
    try:
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension.with_options(no_python_abi_suffix=True)
    except ImportError:
        # Fallback to standard build_ext if PyTorch not available
        return build_ext


def get_extension_class():
    """Get appropriate Extension class."""
    try:
        from torch.utils.cpp_extension import CppExtension
        return CppExtension
    except ImportError:
        from setuptools import Extension
        return Extension


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torch_remote extension
        for path in (ROOT_DIR / "torch_remote").glob("**/*.so"):
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        vc_version = os.getenv("VCToolsVersion", "")
        if vc_version.startswith("14.16."):
            CXX_FLAGS = ["/sdl"]
        else:
            CXX_FLAGS = ["/sdl", "/permissive-"]
    elif platform.machine() == "s390x":
        # no -Werror on s390x due to newer compiler
        CXX_FLAGS = {"cxx": ["-g", "-Wall"]}
    else:
        CXX_FLAGS = {"cxx": ["-g", "-Wall", "-Werror"]}

    sources = list(CSRC_DIR.glob("*.cpp"))

    # Use appropriate Extension class based on PyTorch availability
    ExtensionClass = get_extension_class()
    ext_modules = [
        ExtensionClass(
            name="torch_remote._C",
            sources=sorted(str(s) for s in sources),
            include_dirs=[str(CSRC_DIR)],
            extra_compile_args=CXX_FLAGS,
        )
    ]

    setup(
        name=PACKAGE_NAME,
        version=version,
        author="PyTorch Remote Extension",
        description="Remote GPU cloud execution extension for PyTorch supporting multiple providers",
        packages=find_packages(exclude=("test",)) + ["_torch_remote_modal"],
        package_data={
            PACKAGE_NAME: [
                "*.dll", "*.dylib", "*.so"  # Binary extensions
            ]
        },
        install_requires=[
            "torch>=2.0.0",
            "modal>=0.60.0",
            "numpy",
        ],
        extras_require={
            "runpod": ["runpod>=1.0.0"],  # Future provider support
            "all": ["runpod>=1.0.0"],
        },
        ext_modules=ext_modules,
        python_requires=">=3.8",
        cmdclass={
            "build_ext": get_build_ext_class(),
            "clean": clean,
        },
    )