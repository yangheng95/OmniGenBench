# -*- coding: utf-8 -*-
# file: OmniGenomeRNADesign.py
# time: 20:24 08/05/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import argparse
import json
import os
import random

import autocuda
import numpy as np
import torch

from transformers import AutoModelForMaskedLM, AutoTokenizer

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import ViennaRNA
from scipy.spatial.distance import hamming

def random_bp_span(bp_span):
    # bp = random.choices([bp_span, bp_span - random.randint(0, 20)], k=1)[0]
    bp = random.choice(list(range(max(0, bp_span - 50), min(bp_span + 50, 400))))
    return bp

def longest_bp_span(structure):
    # 使用栈来跟踪括号的匹配
    stack = []
    max_span = 0  # 初始化最长span为0

    # 遍历结构中的每个字符
    for i, char in enumerate(structure):
        if char == '(':
            # 对于左括号，我们将其索引压入栈
            stack.append(i)
        elif char == ')':
            # 对于右括号，我们从栈中弹出最近的左括号
            if stack:
                left_index = stack.pop()
                # 计算当前碱基对的span
                current_span = i - left_index
                # 更新最长的span
                max_span = max(max_span, current_span)

    return max_span

def predict_structure_single(sequence, bp_span=-1):
    """Predicts the RNA structure for a single sequence."""

    md = ViennaRNA.md()
    # md.max_bp_span = bp_span
    md.max_bp_span = max(random_bp_span(bp_span), 400)
    fc = ViennaRNA.fold_compound(sequence, md)
    (ss, mfe) = fc.mfe()
    # (ss_pf, mfe_pf) = fc.pf()
    return ss, mfe


def predict_structure(sequences, bp_span=-1):
    """Predicts structures for multiple sequences using multithreading."""

    return [predict_structure_single(sequence, bp_span) for sequence in sequences]

def genetic_algorithm_for_rna_design(structure, **kwargs):
    mutation_ratio = kwargs.get("mutation_ratio", 0.1)
    num_population = kwargs.get("num_population", 100)
    num_generation = kwargs.get("num_generation", 50)
    # model = "anonymous8/OmniGenome-186M"
    model = kwargs.get("model", None)
    # model = "anonymous8/OmniGenome-52M"
    device = autocuda.auto_cuda()
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForMaskedLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device).to(torch.float16)
    # print(model)
    import tqdm
    histories = []
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
        next_generation = crossover(population)
        next_generation = mlm_mutate(
            next_generation,
            structure,
            model=model,
            tokenizer=tokenizer,
            mutation_ratio=mutation_ratio,
        )
        next_generation = evaluate_structure_fitness(next_generation, structure)[:num_population]
        # print(next_generation[0])
        candidate_sequences = [seq for seq, bp_span, score, mfe in next_generation if score == 0]
        histories.append(next_generation)
        if candidate_sequences:
            return candidate_sequences, histories

        population = [(seq, bp_span) for seq, bp_span, score, mfe in next_generation if seq not in observed]
        observed.update([seq for seq, bp_span, score, mfe in next_generation])
        # next_generation += population[:num_population//10]
        next_generation = population
        # print("population size:", len(next_generation))
    return population[0][0], histories


def init_population(structure, num_population, model, tokenizer):
    population = []
    mlm_inputs = []
    for _ in range(num_population):
        masked_sequence = [
            random.choice(["G", "C", "<mask>"]) for _ in range(len(structure))
        ]
        masked_sequence_str = "".join(masked_sequence)
        mlm_inputs.append(f"{masked_sequence_str}<eos>{''.join(structure)}")

    outputs = mlm_predict(mlm_inputs, structure, model, tokenizer)

    for i in range(len(outputs)):
        sequence = tokenizer.convert_ids_to_tokens(outputs[i].tolist())
        fixed_sequence = [
            # x if x in "AGCT" else random.choice(["A", "T", "G", "C"])
            x
            if x in "AGCT" and y == "$"
            else (y if y and y != "$" else random.choice(["A", "T", "G", "C"]))
            for x, y in zip(sequence, list(mlm_inputs[i].replace("<mask>", "$")))
        ]
        bp_span = random_bp_span(len(structure))

        population.append(("".join(fixed_sequence), bp_span))

    return population


def mlm_mutate(population, structure, model, tokenizer, mutation_ratio):
    def mutate(sequence, mutation_rate):
        sequence = np.array(list(sequence), dtype=np.str_)
        probability_matrix = np.full(sequence.shape, mutation_rate)
        masked_indices = np.random.rand(*sequence.shape) < probability_matrix
        sequence[masked_indices] = "$"
        mut_seq = "".join(sequence.tolist()).replace("$", "<mask>")
        return mut_seq
    random.seed(random.randint(0, 99999999))
    # Initialize lists to store population data and inputs for masked language model
    mlm_inputs = []
    masked_sequences = []
    population_size = len(population)
    top_10_percent_index = population_size // 1
    for i in range(population_size):
        sequence, bp_span = population[random.choice(list(range(top_10_percent_index)))]
        masked_sequence = mutate(sequence, mutation_ratio)
        masked_structure = structure
        masked_sequences.append(masked_sequence)
        mlm_inputs.append(f"{masked_sequence}<eos>{''.join(masked_structure)}")

    # # Iterate over the number of individuals in the population
    # for sequence, bp_span in population:
    #     masked_sequence = mutate(sequence, mutation_ratio)
    #     masked_structure = structure
    #     masked_sequences.append(masked_sequence)
    #     mlm_inputs.append(f"{masked_sequence}<eos>{''.join(masked_structure)}")

    # Call a function to predict outputs using the masked language model
    outputs = mlm_predict(mlm_inputs, structure, model, tokenizer)

    mut_population = []

    # Decode the mlm outputs and construct the initial population
    for i, (seq, bp_span) in zip(range(len(outputs)), population):
        sequence = tokenizer.convert_ids_to_tokens(outputs[i].tolist())
        fixed_sequence = [
            # x if x in "AGCT" else random.choice(["A", "T", "G", "C"])
            x
            if x in "AGCT" and y == "$"
            else (y if y and y != "$" else random.choice(["A", "T", "G", "C"]))
            for x, y in zip(sequence, list(masked_sequences[i].replace("<mask>", "$")))
        ]
        bp_span = random_bp_span(bp_span)
        mut_population.append(("".join(fixed_sequence), bp_span))

    return mut_population


def crossover(population, num_points=3):
    _population = population
    population = [seq for seq, _ in population]
    population_size = len(population)
    sequence_length = len(population[0])
    top_10_percent_index = population_size // 1

    # Convert population to a numpy array for faster indexing
    population_array = np.array(population)

    # Select parents for crossover
    parent_indices = np.random.choice(top_10_percent_index, (population_size, 2))
    parent1_array = np.array([list(seq) for seq in population_array[parent_indices[:, 0]]])
    parent2_array = np.array([list(seq) for seq in population_array[parent_indices[:, 1]]])

    # Determine crossover points for all pairs
    crossover_points = np.sort(np.random.randint(1, sequence_length, size=(population_size, num_points)), axis=1)

    # Initialize masks for slicing parent sequences
    masks = np.zeros((population_size, sequence_length), dtype=bool)

    # Create masks for slicing parent sequences
    for i in range(population_size):
        last_point = 0
        for j in range(num_points):
            masks[i, last_point:crossover_points[i, j]] = (j % 2 == 0)
            last_point = crossover_points[i, j]
        masks[i, last_point:] = (num_points % 2 == 0)

    # Generate children by applying masks
    child1_array = np.where(masks, parent1_array, parent2_array)
    child2_array = np.where(masks, parent2_array, parent1_array)

    # Flatten children arrays into sequences
    children_sequences = [("".join(child1_array[i]), _population[i][1]) for i in range(population_size)] + \
                         [("".join(child2_array[i]), _population[i][1]) for i in range(population_size)]

    return children_sequences


def non_dominated_sorting(scores, mfe_values):
    num_solutions = len(scores)
    domination_count = [0] * num_solutions
    dominated_solutions = [[] for _ in range(num_solutions)]
    fronts = [[]]

    for p in range(num_solutions):
        for q in range(num_solutions):
            if (scores[p] < scores[q] and mfe_values[p] < mfe_values[q]):
                dominated_solutions[p].append(q)
            elif (scores[q] < scores[p] and mfe_values[q] < mfe_values[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if not fronts[-1]:  # Ensure the last front is not empty before removing
        fronts.pop(-1)

    return fronts


def evaluate_structure_fitness(sequences, structure):

    structures_mfe = []
    with ProcessPoolExecutor() as executor:
        for sequence, bp_span in sequences:
            structures_mfe += list(executor.map(predict_structure_single, [sequence], [bp_span]))
    mfe_values = []
    sorted_population = []
    for (seq, bp_span), (ss, mfe) in zip(sequences, structures_mfe):
        mfe_values.append(mfe)
        score = hamming(list(structure), list(ss))
        sorted_population.append((seq, bp_span, score, mfe))

    # sorted_population = sorted(sorted_population, key=lambda x: x[3])[:2*len(sequences)]
    # sorted_population = sorted(sorted_population, key=lambda x: x[2])[:len(sequences)]

    # Perform non-dominated sorting
    fronts = non_dominated_sorting(
        [x[2] for x in sorted_population],
        [x[3] for x in sorted_population],
    )
    # print(structure)

    sorted_population = select_next_generation(sorted_population, fronts)
    sorted_population = sorted(sorted_population, key=lambda x: x[2])

    # print(np.mean([x[1] for x in sorted_population]))
    # print(np.mean([x[2] for x in sorted_population]))
    return sorted_population


def select_next_generation(next_generation, fronts):
    sorted_population = []
    for front in fronts:
        front_population = [next_generation[i] for i in front]
        sorted_population.extend(front_population)
        if len(sorted_population) >= len(next_generation):
            break

    # Truncate to the size of the population if necessary
    next_generation = sorted_population[:len(next_generation)]
    return next_generation


def mlm_predict(mlm_inputs, structure, model, tokenizer):
    batch_size = 16
    all_outputs = []
    from transformers import set_seed

    set_seed(random.randint(0, 99999999), deterministic=False)

    with torch.no_grad():
        for i in range(0, len(mlm_inputs), batch_size):
            batch_mlm_inputs = tokenizer(
                mlm_inputs[i: i + batch_size],
                padding=False,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            )
            batch_mlm_inputs = {key: value.to(torch.int64).to(model.device) for key, value in batch_mlm_inputs.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**batch_mlm_inputs)[0]
            outputs = outputs.argmax(dim=-1)
            all_outputs.append(outputs)
            del batch_mlm_inputs
            del outputs
    outputs = torch.cat(all_outputs, dim=0)
    # outputs[outputs == 7] = 9  # convert all T to U for RNA sequence
    return outputs[:, 1: 1 + len(structure)]


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    observed = set()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="The model name or path")
    parser.add_argument("--structure", default="(((((((..((((......)))).(((((.......)))))...((((..)))))))))))....", type=str, help="The target RNA structure")
    parser.add_argument("--sequence", default="GCUGUGUUAGUAUAAAGUAAUAUAUGUGAUUUCUAAUCAUGGGAUCCUUUAGGGACGUAGUACCA", type=str, help="Reference sequence")
    parser.add_argument("--mutation_ratio", type=float, default=0.5, help="The mutation ratio")
    parser.add_argument("--num_population", type=int, default=100, help="The number of population")
    parser.add_argument("--num_generation", type=int, default=100, help="The number of generation")

    args = parser.parse_args()

    structure = args.structure
    sequence = args.sequence
    mutation_ratio = args.mutation_ratio
    num_population = args.num_population
    num_generation = args.num_generation

    if os.path.exists("solved_sequences.json"):
        solved_sequences = json.load(open("solved_sequences.json", "r"))
    else:
        solved_sequences = {}

    if not (structure in solved_sequences and isinstance(solved_sequences[structure]["best_sequence"], list)):
        best_sequence, histories = genetic_algorithm_for_rna_design(
            structure,
            mutation_ratio=mutation_ratio,
            num_population=num_population,
            num_generation=num_generation,
            model=args.model,
        )

        if os.path.exists("solved_sequences.json"):
            solved_sequences = json.load(open("solved_sequences.json", "r"))
        else:
            solved_sequences = {}

        solved_sequences[structure] = {
            "sequence": sequence,
            "structure": structure,
            "best_sequence": best_sequence,
            # "histories": histories,
        }
        json.dump(solved_sequences, open("solved_sequences.json", "w"))
    else:
        print(f"Structure {structure} has been solved.")

    acc = 0
    total = 0
    if os.path.exists("solved_sequences.json"):
        solved_sequences = json.load(open("solved_sequences.json", "r"))
        for k, v in solved_sequences.items():
            total += 1
            if isinstance(v['best_sequence'], list):
                acc += 1
    print(f"Accuracy: {acc / total * 100}% Total: {total}")
