from setuptools import find_packages, setup

setup(
    name="torch_remote_execution",
    version="0.1.0",
    author="PyTorch Remote Extension",
    description="Private package for torch_remote execution supporting multiple cloud providers (do not install directly)",
    packages=find_packages(),
    install_requires=[
        "modal>=0.60.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
    # Mark as private - not intended for direct installation
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)