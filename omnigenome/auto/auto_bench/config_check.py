# -*- coding: utf-8 -*-
# file: config_verification.py
# time: 02/11/2022 17:05
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

one_shot_messages = set()


def config_check(args):
    """
    Performs a basic check on the configuration arguments.

    This function can be expanded to include more complex validation logic
    for the benchmark configuration.

    :param args: A dictionary of configuration arguments.
    :raises RuntimeError: If a configuration check fails.
    """
    try:
        if "use_amp" in args:
            assert args["use_amp"] in {True, False}
        # if "patience" in args:
        #     assert args["patience"] > 0

    except AssertionError as e:
        raise RuntimeError(
            "Exception: {}. Some parameters are not valid, please see the main example.".format(
                e
            )
        )
