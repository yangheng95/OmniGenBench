# -*- coding: utf-8 -*-
# file: setup.py
# time: 14:54 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

from pathlib import Path
from setuptools import setup, find_packages

# Avoid importing the package at build time to prevent executing heavy imports
# and side effects in omnigenbench/__init__.py before dependencies are installed.
def read_version_from_init() -> str:
    init_path = Path(__file__).parent / "omnigenbench" / "__init__.py"
    text = init_path.read_text(encoding="utf8")
    for line in text.splitlines():
        if line.startswith("__version__"):
            # Expected format: __version__ = "x.y.z"
            import re

            match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", line)
            if match:
                return match.group(1)
    raise RuntimeError("Unable to find __version__ in omnigenbench/__init__.py")

cwd = Path(__file__).parent
long_description = (cwd / "README.MD").read_text(encoding="utf8")

extras = {
    "dev": [
        "dill",
        "pytest",
    ]
}

# This is the main setup.py - it will build omnigenbench by default
# Use setup_omnigenome.py and setup_omnigenbench.py for separate builds
setup(
    name="omnigenbench",
    version=read_version_from_init(),
    description="OmniGenoBench: A comprehensive toolkit for genome analysis.",
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
    packages=find_packages(include=["omnigenbench", "omnigenbench.*", "omnigenome", "omnigenome.*"]),
    entry_points={
        "console_scripts": [
            "ogb=omnigenbench.cli.ogb_cli:main",
            # Legacy commands for backward compatibility
            "autobench=omnigenbench.auto.auto_bench.auto_bench_cli:run_bench",
            "autotrain=omnigenbench.auto.auto_train.auto_train_cli:run_train",
            "autoinfer=omnigenbench.cli.autoinfer_cli:main",
        ],
    },
    install_requires=[
        "findfile>=2.0.0",
        "autocuda>=0.16",
        "metric-visualizer>=0.9.6",
        "termcolor",
        "gitpython",
        "torch>=2.6.0",
        "pandas",
        "viennarna",
        "scikit-learn",
        "accelerate",
        "transformers>=4.46.0",
        "packaging",
        "peft",
        "dill",
        "accelerate",
        "plotly",
        "logomaker",
        "matplotlib",
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
