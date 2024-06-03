# -*- coding: utf-8 -*-
# file: OmniGenomeRNADesign.py
# time: 20:24 08/05/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import autocuda
import numpy as np
import torch

from transformers import AutoModelForMaskedLM, AutoTokenizer
import requests


def predict_structure(sequence):
    import ViennaRNA

    if isinstance(sequence, list):
        structures = [
            ViennaRNA.fold(sequence.replace("T", "U"))[0] for sequence in sequence
        ]
        return structures
    else:
        return ViennaRNA.fold(sequence)[0]


def genetic_algorithm_for_rna_design(structure, **kwargs):
    mutation_ratio = kwargs.get("mutation_ratio", 0.5)
    num_population = kwargs.get("num_population", 100)
    num_generation = kwargs.get("num_generation", 50)
    puzzle_id = kwargs.get("puzzle_id", 0)
    model = "anonymous8/OmniGenome-186M"
    # model = "anonymous8/OmniGenome-52M"
    device = autocuda.auto_cuda()
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForMaskedLM.from_pretrained(model)
    model.to(device)

    import tqdm

    population = init_population(
        structure,
        num_population,
        model=model,
        tokenizer=tokenizer,
    )
    population = mlm_mutate(
        population,
        structure,
        model=model,
        tokenizer=tokenizer,
        mutation_ratio=mutation_ratio,
    )
    for generation_id in tqdm.tqdm(
        range(num_generation), desc="Designing RNA Sequence"
    ):
        population_fitness = evaluate_structure_fitness(
            population,
            structure,
        )[:num_population]
        population = sorted(zip(population, population_fitness), key=lambda x: x[1])[
            :num_population
        ]
        population = [x[0] for x in population]
        next_generation = population  # Elitism
        next_generation += mlm_mutate(
            next_generation,
            structure,
            model=model,
            tokenizer=tokenizer,
            mutation_ratio=mutation_ratio,
        )
        # next_generation += crossover(next_generation, structure, model=model, tokenizer=tokenizer)

        fitness_values = evaluate_structure_fitness(
            next_generation,
            structure,
        )
        next_generation = sorted(
            zip(next_generation, fitness_values), key=lambda x: x[1]
        )

        candidate_sequences = []
        for sequence, fitness in next_generation:
            if fitness == 0:
                candidate_sequences.append(sequence)
            else:
                break
        if candidate_sequences:
            del model, tokenizer
            torch.cuda.empty_cache()
            # print(candidate_sequences)
            return candidate_sequences, puzzle_id
        print(
            f"Generation {generation_id}: {next_generation[0][0]} with fitness {next_generation[0][1]}"
        )
        population = [x[0] for x in next_generation[:num_population]]
    del model, tokenizer
    torch.cuda.empty_cache()
    return population[0], puzzle_id


def init_population(structure, num_population, model, tokenizer):
    population = []
    mlm_inputs = []

    for _ in range(num_population):
        masked_sequence = [
            random.choice(["A", "G", "C", "T", "<mask>"]) for _ in range(len(structure))
        ]
        masked_sequence_str = "".join(masked_sequence)
        mlm_inputs.append(f"{masked_sequence_str}<eos>{''.join(structure)}")

    outputs = mlm_predict(mlm_inputs, structure, model, tokenizer)

    for i in range(len(outputs)):
        sequence = tokenizer.convert_ids_to_tokens(outputs[i].tolist())
        fixed_sequence = [
            # x if x in "AGCTNU" else random.choice(["A", "T", "G", "C"])
            x
            if x in "AGCTNU" and y == "$"
            else (y if y and y != "$" else random.choice(["A", "T", "G", "C"]))
            for x, y in zip(sequence, list(mlm_inputs[i].replace("<mask>", "$")))
        ]
        population.append("".join(fixed_sequence))

    return population


def mlm_mutate(population, structure, model, tokenizer, mutation_ratio):
    def mutate(sequence, mutation_rate):
        sequence = np.array(list(sequence), dtype=np.str_)
        probability_matrix = np.full(sequence.shape, mutation_rate)
        masked_indices = np.random.rand(*sequence.shape) < probability_matrix
        sequence[masked_indices] = "$"
        mut_seq = "".join(sequence.tolist()).replace("$", "<mask>")
        return mut_seq

    def mutate_with_spans_mask(sequence, mutation_rate):
        """使用numpy一次性对多个span应用mask突变"""
        sequence = np.array(list(sequence), dtype=np.str_)
        n = len(sequence)
        span_length = random.randint(1, max(int(len(sequence) * mutation_rate), 3))
        num_spans = random.randint(1, max(int(len(sequence) * mutation_rate), 3))
        start_indices = np.random.choice(n - span_length + 1, num_spans, replace=False)
        masks = [
            np.arange(span_length // span_length) + start_indices
            for start_indices in start_indices
        ]
        for mask in masks:
            sequence[mask] = "$"
        return "".join(sequence).replace("$", "<mask>")

    # Initialize lists to store population data and inputs for masked language model
    mlm_inputs = []
    masked_sequences = []
    # Iterate over the number of individuals in the population
    for sequence in population:
        # mutation_ratio = random.uniform(0.1, 0.9)
        # Create a sequence by randomly choosing nucleotides or a mask token for each position in the structure
        masked_sequence = mutate(sequence, mutation_ratio)
        # masked_sequence = mutate_with_spans_mask(sequence, mutation_ratio)
        masked_sequences.append(masked_sequence)
        mlm_inputs.append(f"{masked_sequence}<eos>{''.join(structure)}")

    # Call a function to predict outputs using the masked language model
    outputs = mlm_predict(mlm_inputs, structure, model, tokenizer)

    mut_population = []

    # Decode the mlm outputs and construct the initial population
    for i in range(len(outputs)):
        sequence = tokenizer.convert_ids_to_tokens(outputs[i].tolist())
        fixed_sequence = [
            # x if x in "AGCTNU" else random.choice(["A", "T", "G", "C"])
            x
            if x in "AGCTNU" and y == "$"
            else (y if y and y != "$" else random.choice(["A", "T", "G", "C"]))
            for x, y in zip(sequence, list(masked_sequences[i].replace("<mask>", "$")))
        ]
        mut_population.append("".join(fixed_sequence))

    return mut_population


def crossover(population, structure, model, tokenizer):
    crossover_population = []
    batch_crossover_inputs = []
    postions = []
    for i in range(len(population)):
        parent1, parent2 = random.choices(population, k=2)
        pos = random.randint(1, len(parent1) - 1)
        postions.append(pos)
        child1 = parent1[:pos] + "<mask>" * len(parent2[pos:])
        child2 = "<mask>" * len(parent1[:pos]) + parent2[pos:]
        batch_crossover_inputs.append(f"{child1}<eos>{structure}")
        batch_crossover_inputs.append(f"{child2}<eos>{structure}")

    outputs = mlm_predict(batch_crossover_inputs, structure, model, tokenizer)

    for i in range(len(outputs)):
        sequence = tokenizer.convert_ids_to_tokens(outputs[i].tolist())
        fixed_sequence = [
            # x if x in "AGCTNU" else random.choice(["A", "T", "G", "C"])
            x
            if x in "AGCTNU" and y == "$"
            else (y if y and y != "$" else random.choice(["A", "T", "G", "C"]))
            for x, y in zip(
                sequence, list(batch_crossover_inputs[i].replace("<mask>", "$"))
            )
        ]
        if i % 2 == 0:
            sequence = (
                population[i // 2][: postions[i // 2]]
                + "".join(fixed_sequence)[postions[i // 2] :]
            )
        else:
            sequence = (
                "".join(fixed_sequence)[: postions[i // 2]]
                + population[i // 2][postions[i // 2] :]
            )
        crossover_population.append(sequence)
        # crossover_population.append("".join(fixed_sequence))

    return crossover_population


def evaluate_structure_fitness(sequences, structure):
    structures = []
    for i in range(0, len(sequences), 10):
        structures += predict_structure(sequences[i : i + 10])
    # structures, mfe_values = zip(*structures)
    fitness_values = []
    for predicted_structure in structures:
        scores = []
        for i in range(len(predicted_structure)):
            if predicted_structure[i] == structure[i]:
                scores.append(1)
            else:
                scores.append(0)

        score = 1 - sum(scores) / len(structure)
        fitness_values.append(score)
    return fitness_values


def mlm_predict(mlm_inputs, structure, model, tokenizer):
    batch_size = 4
    all_outputs = []
    from transformers import set_seed

    set_seed(random.randint(0, 99999999), deterministic=False)

    with torch.no_grad():
        for i in range(0, len(mlm_inputs), batch_size):
            batch_mlm_inputs = tokenizer(
                mlm_inputs[i : i + batch_size],
                padding=False,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            )
            batch_mlm_inputs = batch_mlm_inputs.to(model.device)
            outputs = model(**batch_mlm_inputs)[0]
            outputs = outputs.argmax(dim=-1)
            all_outputs.append(outputs)
            del batch_mlm_inputs
            del outputs
    outputs = torch.cat(all_outputs, dim=0)
    # outputs[outputs == 7] = 9  # convert all T to U for RNA sequence
    return outputs[:, 1 : 1 + len(structure)]


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    structures = []
    sequences = []
    solved_sequences = {}

    with open("eterna100_vienna2.txt", encoding="utf8", mode="r") as f:
        for line in f.readlines()[1:]:
            parts = line.split("\t")
            structures.append(parts[4].strip())
            sequences.append(parts[5].strip())

    structures = structures[:]
    sequences = sequences[:]
    outputs = []
    pred_count = 0
    acc_count = 0
    # for i, structure in enumerate(structures):
    #     best_sequence, puzzle_id = (
    #         genetic_algorithm_for_rna_design(structure, **dict(
    #             mutation_ratio=0.5,
    #             num_population=50,
    #             num_generation=50,
    #             puzzle_id=i
    #         )))
    #     pred_count += 1
    #     if isinstance(best_sequence, list):
    #         acc_count += 1
    #         print("Best sequence found:", best_sequence)
    #     else:
    #         best_sequence = best_sequence
    #         print("Not Found")
    #     print(f"Sum: {pred_count} Accuracy:", acc_count / pred_count * 100)
    #     solved_sequences[puzzle_id] = (sequences[puzzle_id], structure, best_sequence)
    #     print(solved_sequences)
    #
    # with open("eterna100_vienna2.txt.result", encoding="utf8", mode="w") as fw:
    #     for i, (sequence, target_structure, best_sequence) in solved_sequences.items():
    #         fw.write(f"{i}\t{sequence}\t{target_structure}\t{best_sequence}\n")

    with ProcessPoolExecutor(os.cpu_count()) as executor:
        for i, target_structure in enumerate(structures):
            time.sleep(3)
            outputs.append(
                executor.submit(
                    genetic_algorithm_for_rna_design,
                    target_structure,
                    **dict(
                        mutation_ratio=0.5,
                        num_population=100,
                        num_generation=50,
                        puzzle_id=i,
                    ),
                )
            )

        for result in as_completed(outputs):
            pred_count += 1
            best_sequence, puzzle_id = result.result()
            if isinstance(best_sequence, list):
                acc_count += 1
                print("Best sequence found:", best_sequence)
            else:
                best_sequence = best_sequence
                print("Not Found")
            print(f"Sum: {pred_count} Accuracy:", acc_count / pred_count * 100)
            solved_sequences[puzzle_id] = (
                sequences[puzzle_id],
                target_structure,
                best_sequence,
            )

        with open("eterna100_vienna2.txt.result", encoding="utf8", mode="w") as fw:
            for i, (
                sequence,
                target_structure,
                best_sequence,
            ) in solved_sequences.items():
                fw.write(f"{i}\t{sequence}\t{target_structure}\t{best_sequence}\n")
