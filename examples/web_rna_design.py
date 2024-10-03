# -*- coding: utf-8 -*-
# file: test.py
# time: 18:43 14/08/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import os
import random
import autocuda
import numpy as np
import torch
import gradio as gr

from transformers import AutoModelForMaskedLM, AutoTokenizer

from concurrent.futures import ThreadPoolExecutor
import ViennaRNA as RNA
import matplotlib.pyplot as plt

def predict_structure_single(sequence):
    """Predicts the RNA structure for a single sequence."""
    return RNA.fold(sequence.replace("T", "U"))[0]
#
# def predict_structure(sequences):
#     """Predicts structures for multiple sequences using multithreading."""
#     if isinstance(sequences, list):
#         with ThreadPoolExecutor() as executor:
#             # Map the predict_structure_single function to each sequence
#             structures = list(executor.map(predict_structure_single, sequences))
#         return structures
#     else:
#         # Single sequence case
#         return predict_structure_single(sequences)


def predict_structure(sequences):
    """Predicts structures for multiple sequences using multithreading."""
    if isinstance(sequences, list):
        with ThreadPoolExecutor() as executor:
            # Map the predict_structure_single function to each sequence
            structures = list(executor.map(predict_structure_single, sequences))
        return structures
    else:
        # Single sequence case
        return predict_structure_single(sequences)



def genetic_algorithm_for_rna_design(structure, visualize_generation_callback=None, **kwargs):
    mutation_ratio = kwargs.get("mutation_ratio", 0.5)
    num_population = kwargs.get("num_population", 500)
    num_generation = kwargs.get("num_generation", 10)
    random.seed(random.randint(0, 99999999))
    puzzle_id = kwargs.get("puzzle_id", 0)
    model = "anonymous8/OmniGenome-186M"
    device = autocuda.auto_cuda()
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForMaskedLM.from_pretrained(model, trust_remote_code=True)
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
        mutation_ratio = mutation_ratio
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

        fitness_values = evaluate_structure_fitness(
            next_generation,
            structure,
        )
        next_generation = sorted(
            zip(next_generation, fitness_values), key=lambda x: x[1]
        )

        # 每一代输出最优序列
        best_sequence = next_generation[0][0]
        best_fitness = next_generation[0][1]

        if visualize_generation_callback:
            visualize_generation_callback(best_sequence)

        candidate_sequences = []
        for sequence, fitness in next_generation:
            if fitness == 0:
                candidate_sequences.append(sequence)
            else:
                break
        if candidate_sequences:
            del model, tokenizer
            torch.cuda.empty_cache()
            return f"Success!    best_sequence: {best_sequence}, best_fitness: {best_fitness} "

        population = [x[0] for x in next_generation[:num_population]]
    del model, tokenizer
    torch.cuda.empty_cache()
    return f"Failed!    best_sequence: {best_sequence}, best_fitness: {best_fitness}"


def visualize_generation(structure, sequence, generation_id, fitness):
    plt.figure(figsize=(6, 6))
    rna_structure = RNA.fold(sequence.replace("T", "U"))[0]
    RNA.svg_rna_plot(sequence, rna_structure, "predicted_structure.svg")
    img = plt.imread("predicted_structure.svg")
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Generation {generation_id} with Fitness {fitness:.4f}')
    plt.show()
    os.remove("predicted_structure.svg")


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
            x if x in "AGCT" and y == "$" else (y if y and y != "$" else random.choice(["A", "T", "G", "C"]))
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

    mlm_inputs = []
    masked_sequences = []
    for sequence in population:
        masked_sequence = mutate(sequence, mutation_ratio)
        masked_sequences.append(masked_sequence)
        mlm_inputs.append(f"{masked_sequence}<eos>{''.join(structure)}")

    outputs = mlm_predict(mlm_inputs, structure, model, tokenizer)

    mut_population = []

    for i in range(len(outputs)):
        sequence = tokenizer.convert_ids_to_tokens(outputs[i].tolist())
        fixed_sequence = [
            x if x in "AGCT" and y == "$" else (y if y and y != "$" else random.choice(["A", "T", "G", "C"]))
            for x, y in zip(sequence, list(masked_sequences[i].replace("<mask>", "$")))
        ]
        mut_population.append("".join(fixed_sequence))

    return mut_population


def evaluate_structure_fitness(sequences, structure):
    structures = []
    for i in range(0, len(sequences), 10):
        structures += predict_structure(sequences[i: i + 10])
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
                mlm_inputs[i: i + batch_size],
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
    return outputs[:, 1: 1 + len(structure)]


# Gradio interface functions
def gradio_visualize_generation(sequence):
    rna_structure = RNA.fold(sequence.replace("T", "U"))[0]
    RNA.svg_rna_plot(sequence, rna_structure, f"predicted_structure.svg")


def find_invalid_ss_positions(rna_strct):
    left_brackets = []  # 存储左括号的位置
    right_brackets = []  # 存储未匹配的右括号的位置
    for i, char in enumerate(rna_strct):
        if char == "(":
            left_brackets.append(i)
        elif char == ")":
            if left_brackets:
                left_brackets.pop()  # 找到匹配的左括号，从列表中移除
            else:
                right_brackets.append(i)  # 没有匹配的左括号，记录右括号的位置
    return left_brackets + right_brackets



def repair_rna_structure(rna_sequence, invalid_struct):
    try:
        invalid_ss_positions = find_invalid_ss_positions(invalid_struct)
        for pos_idx in invalid_ss_positions:
            if invalid_struct[pos_idx] == "(":
                invalid_struct = (
                    invalid_struct[:pos_idx] + "." + invalid_struct[pos_idx + 1 :]
                )
            else:
                invalid_struct = (
                    invalid_struct[:pos_idx] + "." + invalid_struct[pos_idx + 1 :]
                )

        best_pred_struct = invalid_struct
        RNA.svg_rna_plot(rna_sequence, best_pred_struct, f"predicted_structure.svg")
        return best_pred_struct
    except Exception as e:
        with open("best_pred_struct.svg", "w") as f:
            f.write(
                '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>'
            )
        return "best_pred_struct.svg"


def gradio_genetic_algorithm(structure):
    return genetic_algorithm_for_rna_design(structure, visualize_generation_callback=gradio_visualize_generation)

def reload_image(sequence, structure):
    if structure and sequence:
        RNA.svg_rna_plot(sequence, structure, f"predicted_structure.svg")
    else:
        pass
    return f"true_struct.svg", f"predicted_structure.svg"

def sample_rna_design_puzzle():
    idx = random.randint(0, len(sequences)-1)
    structure = structures[idx]
    sequence = sequences[idx]
    RNA.svg_rna_plot(sequence, structure, f"true_struct.svg")
    return sequence, structure


if __name__ == "__main__":

    benchmark_file = "eterna100_vienna2.txt"
    # benchmark_file = "eterna100_contrafold.txt"
    structures = []
    sequences = []
    solved_sequences = {}

    with open(benchmark_file, encoding="utf8", mode="r") as f:
        for line in f.readlines()[1:]:
            parts = line.split("\t")
            if len(parts[5].strip()) < 200:
                structures.append(parts[4].strip())
                sequences.append(parts[5].strip())

    with gr.Blocks() as demo:
        gr.Markdown("### RNA Design and Visualization")
        structure_text = gr.Textbox(label="Sampled RNA Structure")
        sequence_text = gr.Textbox(label="Reference RNA Sequence")
        status_text = gr.Textbox(label="Status")
        sample_btn = gr.Button("Sample a RNA Design Puzzle")
        sample_btn.click(sample_rna_design_puzzle, inputs=[], outputs=[sequence_text, structure_text])

        with gr.Row():
            true_image = gr.Image(label="Reference RNA Structure Folding")
            image = gr.Image(label="Predicted RNA Structure Folding")

        run_btn = gr.Button("Run RNA Design")
        run_btn.click(gradio_genetic_algorithm, inputs=[structure_text], outputs=[status_text])
        demo.load(reload_image, inputs=[sequence_text, structure_text],
                  outputs=[true_image, image], show_progress=False, every=0.05)

    demo.launch(share=True)
