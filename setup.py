# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import distutils.command.clean
import os
import platform
import shutil
import sys
from pathlib import Path

from setuptools import find_packages, setup

PACKAGE_NAME = "mycelya_torch"
version = "0.1.2"

ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / "mycelya_torch/csrc"

# Minimum required PyTorch version
TORCH_MIN_VERSION = "2.6.0"


def parse_version(version_str):
    """Parse version string into (major, minor) tuple for comparison."""
    try:
        version_str = version_str.split("+")[0]  # Remove +cu118
        return tuple(int(p) for p in version_str.split(".")[:2])
    except (ValueError, AttributeError, IndexError):
        return (0, 0)


def check_pytorch_installation():
    """Check if PyTorch is installed and meets version requirements."""
    try:
        import torch
    except ImportError:
        print(
            "\nERROR: PyTorch is not installed.",
            "\n",
            "\nInstall PyTorch first:",
            "\n  pip install torch",
            "\n",
            "\nThen retry building mycelya-torch\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check PyTorch version
    torch_version = torch.__version__
    torch_version_tuple = parse_version(torch_version)
    min_version_tuple = parse_version(TORCH_MIN_VERSION)

    if torch_version_tuple < min_version_tuple:
        print(
            f"\nERROR: PyTorch {torch_version} is too old (need >= {TORCH_MIN_VERSION})",
            "\n",
            "\nUpgrade PyTorch:",
            "\n  pip install --upgrade torch",
            "\n",
            "\nThen retry building mycelya-torch\n",
            file=sys.stderr,
        )
        sys.exit(1)


# Check PyTorch installation before proceeding
check_pytorch_installation()


def get_build_ext_class():
    """Get PyTorch's BuildExtension class."""
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True)


def get_extension_class():
    """Get PyTorch's CppExtension class."""
    from torch.utils.cpp_extension import CppExtension

    return CppExtension


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove mycelya_torch extension
        for path in (ROOT_DIR / "mycelya_torch").glob("**/*.so"):
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
        CXX_FLAGS = ["-g", "-Wall"]
    else:
        CXX_FLAGS = ["-g", "-Wall", "-Werror"]

    sources = list(CSRC_DIR.glob("*.cpp"))

    # Use appropriate Extension class based on PyTorch availability
    ExtensionClass = get_extension_class()
    ext_modules = [
        ExtensionClass(
            name="mycelya_torch._C",
            sources=sorted(str(s.relative_to(ROOT_DIR)) for s in sources),
            include_dirs=[str(CSRC_DIR.relative_to(ROOT_DIR))],
            extra_compile_args=CXX_FLAGS,
        )
    ]

    setup(
        name=PACKAGE_NAME,
        version=version,
        description="Mycelya: PyTorch extension for transparent remote GPU execution on cloud infrastructure",
        long_description=open(ROOT_DIR / "README.md").read(),
        long_description_content_type="text/markdown",
        author="Mycelya Extension",
        license="AGPL-3.0-or-later",
        url="https://github.com/alyxya/mycelya-torch",
        packages=find_packages(exclude=("tests",)),
        python_requires=">=3.10",
        install_requires=[
            "modal>=1.1.0",
            "numpy",
            "cloudpickle>=3.1.1",
        ],
        extras_require={
            "aws": ["boto3>=1.33.0"],
            "all": ["boto3>=1.33.0"],
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        package_data={
            "mycelya_torch": ["*.dll", "*.dylib", "*.so"],
        },
        ext_modules=ext_modules,
        cmdclass={
            "build_ext": get_build_ext_class(),
            "clean": clean,
        },
    )
