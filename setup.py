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
    # Author details
    author="Yang, Heng",
    author_email="hy345@exeter.ac.uk",
    python_requires=">=3.9",
    packages=find_packages(),
    include_package_data=True,
    exclude_package_date={"": [".gitignore"]},
    license="MIT",
    install_requires=[
        "findfile>=2.0.0",
        "autocuda>=0.16",
        "metric-visualizer>=0.9.6",
        "tqdm",
        "termcolor",
        "gitpython",  # need git installed in your OS
        "torch>=1.0.0",
        "sentencepiece",
        "protobuf<4.0.0",
        "pandas",
        "viennarna",
        "scikit-learn",
    ],
    dependency_links=[
        "git+https://github.com/yangheng95/transformers@add_omnigenome#egg=transformers"
    ],
    extras_require=extras,
)
