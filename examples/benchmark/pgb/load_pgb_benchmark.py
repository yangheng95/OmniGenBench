# -*- coding: utf-8 -*-
# file: load_pgb_benchmark.py
# time: 19:37 25/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import json
import os.path

from datasets import load_dataset

config_list = [
    "poly_a.arabidopsis_thaliana",
    "poly_a.oryza_sativa_indica_group",
    "poly_a.trifolium_pratense",
    "poly_a.medicago_truncatula",
    "poly_a.chlamydomonas_reinhardtii",
    "poly_a.oryza_sativa_japonica_group",
    "splicing.arabidopsis_thaliana_donor",
    "splicing.arabidopsis_thaliana_acceptor",
    "lncrna.m_esculenta",
    "lncrna.z_mays",
    "lncrna.g_max",
    "lncrna.s_lycopersicum",
    "lncrna.t_aestivum",
    "lncrna.s_bicolor",
    "promoter_strength.leaf",
    "promoter_strength.protoplast",
    "terminator_strength.leaf",
    "terminator_strength.protoplast",
    "gene_exp.glycine_max",
    "gene_exp.oryza_sativa",
    "gene_exp.solanum_lycopersicum",
    "gene_exp.zea_mays",
    "gene_exp.arabidopsis_thaliana",
    "chromatin_access.oryza_sativa_MH63_RS2",
    "chromatin_access.setaria_italica",
    "chromatin_access.oryza_sativa_ZS97_RS2",
    "chromatin_access.arabidopis_thaliana",
    "chromatin_access.brachypodium_distachyon",
    "chromatin_access.sorghum_bicolor",
    "chromatin_access.zea_mays",
    "pro_seq.m_esculenta",
]


def download_dataset(config):
    # load and truncate dataset and save to disk
    dataset = load_dataset(
        "InstaDeepAI/plant-genomic-benchmark",
        config,
        # trust_remote_code=True
    )
    if "splicing" in config:
        if not os.path.exists("/".join(config.split("."))):
            os.makedirs("/".join(config.split(".")))
        train_positive = dataset["train"].filter(lambda x: x["label"] == 1)
        train_negative = dataset["train"].filter(lambda x: x["label"] == 0)

        trainset = []
        for i in range(4000):
            trainset.append({**train_positive[i]})
            trainset.append({**train_negative[i]})

        test_positive = dataset["test"].filter(lambda x: x["label"] == 1)
        test_negative = dataset["test"].filter(lambda x: x["label"] == 0)

        with open("/".join(config.split(".")) + "/train.json", "w") as f:
            for line in trainset:
                f.write(json.dumps(line) + "\n")

        testset = []
        for i in range(500):
            testset.append({**test_positive[i]})
            testset.append({**test_negative[i]})

        with open("/".join(config.split(".")) + "/test.json", "w") as f:
            for line in testset:
                f.write(json.dumps(line) + "\n")

        if "valid" in dataset:
            valid_positive = dataset["valid"].filter(lambda x: x["label"] == 1)
            valid_negative = dataset["valid"].filter(lambda x: x["label"] == 0)
            validset = []
            for i in range(500):
                validset.append({**valid_positive[i]})
                validset.append({**valid_negative[i]})

            with open("/".join(config.split(".")) + "/valid.json", "w") as f:
                for line in validset:
                    f.write(json.dumps(line) + "\n")

    else:
        trainset = dataset["train"].select(range(8000))
        testset = dataset["test"].select(range(1000))
        validset = dataset["valid"].select(range(1000)) if "valid" in dataset else None

        if not os.path.exists("/".join(config.split("."))):
            os.makedirs("/".join(config.split(".")))
        trainset.to_json("/".join(config.split(".")) + "/train.json")
        testset.to_json("/".join(config.split(".")) + "/test.json")
        if validset:
            validset.to_json("/".join(config.split(".")) + "/valid.json")
    print(f"{config} done!")


if __name__ == "__main__":
    import multiprocessing

    for config in config_list:
        # p = multiprocessing.Process(target=download_dataset, args=(config,))
        # p.start()
        # for config in config_list:
        #     p.join()
        # print('All done!')
        if "splicing" in config:
            download_dataset(config)
