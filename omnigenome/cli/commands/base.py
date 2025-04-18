# -*- coding: utf-8 -*-
# file: base.py
# time: 19:04 05/02/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
from abc import ABC, abstractmethod


class BaseCommand(ABC):
    @classmethod
    @abstractmethod
    def register_command(cls, subparsers):
        pass

    @classmethod
    def add_common_arguments(cls, parser):
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set the logging level",
        )
        parser.add_argument(
            "--output-dir",
            default="results",
            help="Output directory to save results",
        )
