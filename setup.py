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

from omnigenome import __name__, __version__

cwd = Path(__file__).parent
long_description = (cwd / "README.MD").read_text(encoding="utf8")

extras = {}
extras["dev"] = [
    "dill",
    "pytest",
]

setup(
    name=__name__,
    version=__version__,
    description="OmniGenome: A comprehensive toolkit for genome analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/yangheng95/{__name__}",
    author="Yang, Heng",
    author_email="hy345@exeter.ac.uk",
    python_requires=">=3.9",
    platforms=["Windows", "Linux", "Mac OS-X"],
    packages=find_packages(),
    include_package_data=True,
    exclude_package_date={"": [".gitignore"]},
    license="MIT",
    entry_points={
        "console_scripts": [
            # "omnigenome-bench=omnigenome:bench_command",
            "autobench=omnigenome:run_bench",
        ],
    },
    install_requires=[
        "findfile>=2.0.0",
        "autocuda>=0.16",
        "metric-visualizer>=0.9.6",
        "termcolor",
        "gitpython",
        "torch>=2.0.0",
        "pandas",
        "viennarna",
        "scikit-learn",
        "accelerate",
        "transformers>=4.45.0",
    ],
    extras_require=extras,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)