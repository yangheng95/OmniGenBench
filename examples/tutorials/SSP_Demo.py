# -*- coding: utf-8 -*-
# file: SSP_Demo.py
# time: 17:04 18/04/2025
# author: YANG, HENG <hy345@exeter.ac.uk>
# Description: Display Ground Truth, ViennaRNA prediction, and model prediction in a single view (three images aligned horizontally)）

import os
import time
import base64
import tempfile
from pathlib import Path
import json
import numpy as np
import gradio as gr
import RNA
from omnigenome import ModelHub

# Load the SSP structure prediction model
print("Available files:", os.listdir('..'))
ssp_model = ModelHub.load("../OmniGenome-186M-SSP")

#  Create a temporary directory for storing SVG images
TEMP_DIR = Path(tempfile.mkdtemp())
print(f"Using temporary directory: {TEMP_DIR}")


def ss_validity_loss(rna_strct: str) -> float:
    """
    Compute the structural validity loss: unbalanced brackets / total pairs.
    Helps detect invalid dot-bracket notations.
    """
    left = right = 0
    dots = rna_strct.count('.')
    for c in rna_strct:
        if c == '(':
            left += 1
        elif c == ')':
            if left:
                left -= 1
            else:
                right += 1
        elif c != '.':
            raise ValueError(f"Invalid char {c}")
    return (left + right) / (len(rna_strct) - dots + 1e-8)


def find_invalid_positions(struct: str) -> list:
    """
    Find invalid base-pair positions in dot-bracket string (e.g. unmatched parentheses).
    """
    stack, invalid = [], []
    for i, c in enumerate(struct):
        if c == '(': stack.append(i)
        elif c == ')':
            if stack:
                stack.pop()
            else:
                invalid.append(i)
    invalid.extend(stack)
    return invalid


def generate_svg_datauri(rna_seq: str, struct: str) -> str:
    """
    Generate SVG from RNA sequence and structure, and return Base64-encoded data URI.
    """
    try:
        path = TEMP_DIR / f"{hash(rna_seq+struct)}.svg"
        RNA.svg_rna_plot(rna_seq, struct, str(path))
        time.sleep(0.1)
        svg_bytes = path.read_bytes()
        b64 = base64.b64encode(svg_bytes).decode('utf-8')
    except Exception as e:
        err = ('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200">'
               f'<text x="50" y="100" fill="red">Error: {e}</text></svg>')
        b64 = base64.b64encode(err.encode()).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"


def fold(rna_seq: str, gt_struct: str):
    """
    Display side-by-side structural comparison:
    Ground Truth (if provided), ViennaRNA prediction, and SSP model prediction.
    """
    if not rna_seq.strip():
        return "", "", "", ""
    # Ground Truth (user-provided)
    ground = gt_struct.strip() if gt_struct and gt_struct.strip() else ""
    gt_uri = generate_svg_datauri(rna_seq, ground) if ground else ""

    # ViennaRNA folding prediction
    vienna_struct, vienna_energy = RNA.fold(rna_seq)
    vienna_uri = generate_svg_datauri(rna_seq, vienna_struct)

    # Model prediction
    result = ssp_model.inference(rna_seq)
    pred = "".join(result.get('predictions', []))
    if ss_validity_loss(pred):
        for i in find_invalid_positions(pred):
            pred = pred[:i] + '.' + pred[i+1:]
    pred_uri = generate_svg_datauri(rna_seq, pred)

    # Structural similarity metrics
    match_gt = (sum(a==b for a,b in zip(ground, pred)) / len(ground)) if ground else 0
    match_vienna = sum(a==b for a,b in zip(vienna_struct, pred)) / len(vienna_struct)
    stats = (
        f"GT↔Pred Match: {match_gt:.2%}" + (" | " if ground else "") +
        f"Vienna↔Pred Match: {match_vienna:.2%}"
    )

    # Combine the three SVGs in horizontal layout
    combined = (
        '<div style="display:flex;justify-content:space-around;">'
        f'{f"<div><h4>Ground Truth</h4><img src=\"{gt_uri}\" style=\"max-width:100%;height:auto;\"/></div>" if ground else ""}'
        f'<div><h4>ViennaRNA</h4><img src=\"{vienna_uri}\" style=\"max-width:100%;height:auto;\"/></div>'
        f'<div><h4>Prediction</h4><img src=\"{pred_uri}\" style=\"max-width:100%;height:auto;\"/></div>'
        '</div>'
    )
    return ground, vienna_struct, pred, stats, combined


def sample_rna_sequence():
    """
    Randomly sample an RNA sequence and ground truth structure from the test set.
    """
    try:
        exs = [json.loads(l) for l in open('toy_datasets/Archive2/test.json')]
        ex = exs[np.random.randint(len(exs))]
        return ex['seq'], ex.get('label','')
    except Exception as e:
        return f"Failed to load sample: {e}", ""

# Gradio UI
with gr.Blocks(css="""
.heading {text-align:center;color:#2a4365;}
.controls {display:flex;gap:10px;margin:20px 0;}
.status {padding:10px;background:#f0f4f8;border-radius:4px;white-space:pre;}
""") as demo:
    gr.Markdown("# RNA Structure Prediction Comparison", elem_classes="heading")
    with gr.Row():
        rna_input = gr.Textbox(label="RNA Sequence", lines=3)
        structure_input = gr.Textbox(label="Ground Truth Structure (optional)", lines=3)
    with gr.Row(elem_classes="controls"):
        sample_btn = gr.Button("Sample Sequence")
        run_btn = gr.Button("Predict & Compare", variant="primary")
    stats_out    = gr.Textbox(label="Statistics", interactive=False, elem_classes="status")
    gt_out       = gr.Textbox(label="Ground Truth", interactive=False)
    vienna_out   = gr.Textbox(label="ViennaRNA Structure", interactive=False)
    pred_out     = gr.Textbox(label="Prediction Structure", interactive=False)
    combined_view= gr.HTML(label="Side-by-Side Comparison View")

    run_btn.click(
        fold,
        inputs=[rna_input, structure_input],
        outputs=[gt_out, vienna_out, pred_out, stats_out, combined_view]
    )
    sample_btn.click(
        sample_rna_sequence,
        outputs=[rna_input, structure_input]
    )

    demo.launch(share=True)
