# -*- coding: utf-8 -*-
# file: setup_omnigenbench.py
# time: 14:54 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

from pathlib import Path
from setuptools import setup, find_packages
from omnigenbench import __version__

cwd = Path(__file__).parent
long_description = (cwd / "README.MD").read_text(encoding="utf8")

extras = {
    "dev": [
        "dill",
        "pytest",
    ]
}

setup(
    name="omnigenbench",
    version=__version__,
    description="OmniGenBench: A comprehensive toolkit for genome analysis benchmarking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yangheng95/OmniGenBench",
    author="Yang, Heng",
    author_email="hy345@exeter.ac.uk",
    python_requires=">=3.10",
    platforms=["Windows", "Linux", "Mac OS-X"],
    include_package_data=True,
    exclude_package_data={"": [".gitignore"]},
    license="Apache-2.0",
    packages=find_packages(include=["omnigenbench", "omnigenbench.*"]),
    entry_points={
        "console_scripts": [
            "omnigenbench=omnigenbench:run_bench",
            "omnigenbench-train=omnigenbench:run_train",
        ],
    },
    install_requires=[
        "omnigenome>=0.3.0alpha1",  # Depend on the main package
    ],
    extras_require=extras,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
