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
    """
    Abstract base class for all CLI commands in OmniGenome.

    This class provides a common interface for all command-line interface
    commands in the OmniGenome framework. It defines the structure that
    all command classes must follow, including registration and common
    argument handling.

    Subclasses must implement the `register_command` method to define
    their specific command-line interface and arguments.

    Example:
        >>> class MyCommand(BaseCommand):
        ...     @classmethod
        ...     def register_command(cls, subparsers):
        ...         parser = subparsers.add_parser("mycommand", help="My command")
        ...         parser.add_argument("--input", required=True)
        ...         parser.set_defaults(func=cls.execute)
        ...
        ...     @staticmethod
        ...     def execute(args):
        ...         print(f"Executing with input: {args.input}")
    """

    @classmethod
    @abstractmethod
    def register_command(cls, subparsers):
        """
        Register the command and its arguments with the main parser.

        This abstract method must be implemented by all subclasses to define
        their specific command-line interface, including arguments, help text,
        and default functions.

        Args:
            subparsers: The subparsers object from the main ArgumentParser

        Example:
            >>> parser = argparse.ArgumentParser()
            >>> subparsers = parser.add_subparsers()
            >>> MyCommand.register_command(subparsers)
        """
        pass

    @classmethod
    def add_common_arguments(cls, parser):
        """
        Add common arguments to a command's parser.

        This method adds standard arguments that are common across all
        OmniGenome CLI commands, such as logging level and output directory.

        Args:
            parser: The ArgumentParser for the specific command

        Example:
            >>> parser = argparse.ArgumentParser()
            >>> BaseCommand.add_common_arguments(parser)
        """
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
