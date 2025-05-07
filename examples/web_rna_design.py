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
import tempfile

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


# ---------- Helpers to generate SVG as string ----------

def generate_svg_string(seq: str, struct: str) -> str:
    """Generate an SVG string of the RNA fold."""
    # Create a temporary file for the SVG
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as temp_file:
        temp_path = temp_file.name

    # Generate the SVG file
    RNA.svg_rna_plot(seq, struct, temp_path)

    # Read the SVG file
    with open(temp_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    # Clean up the temporary file
    try:
        os.unlink(temp_path)
    except:
        pass  # In case of any file deletion issues

    return svg_content


def svg_to_html(svg_string: str) -> str:
    """Convert SVG string to HTML code that can be displayed in Gradio HTML component."""
    # Wrap SVG in a div with proper styling
    return f"""
    <div style="display: flex; justify-content: center; margin: 10px 0;">
        {svg_string}
    </div>
    """

# ---------- Gradio callbacks ----------

def sample_rna_design_puzzle():
    idx = random.randrange(len(sequences))
    struct = structures[idx]
    seq = sequences[idx]
    # Generate SVG string and convert to HTML
    svg_str = generate_svg_string(seq, struct)
    html_view = svg_to_html(svg_str)
    # Return sequence, structure, and the HTML
    return seq, struct, html_view


# Store the latest predicted structure HTML
latest_predicted_html = ""


def update_visual(seq, structure):
    global latest_predicted_html
    svg_str = generate_svg_string(seq, structure)
    latest_predicted_html = svg_to_html(svg_str)
    return latest_predicted_html


def run_design(structure):
    global latest_predicted_html
    # Reset the latest HTML
    latest_predicted_html = ""

    # Run the genetic algorithm with the visualization callback
    status = genetic_algorithm_for_rna_design(
        structure,
        visualize_generation_callback=lambda s: update_visual(s, structure),
    )

    # Return the status and the latest HTML
    return status, latest_predicted_html


# ---------- Main & UI setup ----------

if __name__ == "__main__":
    # load benchmark
    structures, sequences = [], []
    with open("eterna100_vienna2.txt", "r", encoding="utf8") as f:
        for line in f.readlines()[1:]:
            cols = line.strip().split("\t")
            if len(cols) > 5 and len(cols[5]) < 200:
                structures.append(cols[4])
                sequences.append(cols[5])

    # build Gradio app
    with gr.Blocks() as demo:
        gr.Markdown("## RNA Design & Visualization Demo")

        with gr.Row():
            seq_in = gr.Textbox(label="RNA Sequence", lines=1)
            struct_in = gr.Textbox(label="Target Structure", lines=1)
            status_out = gr.Textbox(label="Status")

        # HTML components instead of image components
        true_struct_html = gr.HTML(label="True Structure")
        predicted_struct_html = gr.HTML(label="Predicted Structure")

        sample_btn = gr.Button("Sample a RNA Design Puzzle")
        sample_btn.click(
            fn=sample_rna_design_puzzle,
            inputs=[],
            outputs=[seq_in, struct_in, true_struct_html]
        )

        run_btn = gr.Button("Run RNA Design")
        run_btn.click(
            fn=run_design,
            inputs=[struct_in],
            outputs=[status_out, predicted_struct_html]
        )

    demo.launch(share=True)