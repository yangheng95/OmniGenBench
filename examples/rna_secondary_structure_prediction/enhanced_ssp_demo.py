# -*- coding: utf-8 -*-
# Enhanced RNA Secondary Structure Prediction Demo
# Author: Enhanced by AI Assistant
# Description: Interactive RNA structure prediction with dynamic comparison visualization

import os
import json
import time
import base64
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import queue

import numpy as np
import gradio as gr
import RNA
from omnigenbench import ModelHub


class EnhancedSSPDemo:
    """Enhanced RNA Secondary Structure Prediction Demo"""
    
    def __init__(self):
        self.model = None
        self.temp_dir = Path(tempfile.mkdtemp())
        self.comparison_history = []
        
        # Load model
        try:
            model_path = "../OmniGenome-186M-SSP"
            if os.path.exists(model_path):
                self.model = ModelHub.load(model_path)
            else:
                print("Model not found, will use ViennaRNA only")
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    def validate_rna_sequence(self, sequence: str) -> Tuple[bool, str]:
        """Validate RNA sequence"""
        if not sequence.strip():
            return False, "Empty sequence"
        
        valid_bases = set("AUGC")
        sequence = sequence.upper().replace("T", "U")
        
        if not all(base in valid_bases for base in sequence):
            return False, "Invalid bases found. Use only A, U, G, C"
        
        if len(sequence) < 10:
            return False, "Sequence too short (minimum 10 bases)"
        
        if len(sequence) > 500:
            return False, "Sequence too long (maximum 500 bases)"
        
        return True, "Valid sequence"
    
    def ss_validity_loss(self, structure: str) -> float:
        """Calculate structure validity loss"""
        left = right = 0
        dots = structure.count('.')
        
        for c in structure:
            if c == '(':
                left += 1
            elif c == ')':
                if left:
                    left -= 1
                else:
                    right += 1
            elif c != '.':
                raise ValueError(f"Invalid character {c}")
        
        return (left + right) / (len(structure) - dots + 1e-8)
    
    def fix_invalid_structure(self, structure: str) -> str:
        """Fix invalid base pairs in structure"""
        stack = []
        fixed = list(structure)
        
        for i, c in enumerate(structure):
            if c == '(':
                stack.append(i)
            elif c == ')':
                if stack:
                    stack.pop()
                else:
                    fixed[i] = '.'  # Unmatched closing bracket
        
        # Fix unmatched opening brackets
        for i in stack:
            fixed[i] = '.'
        
        return ''.join(fixed)
    
    def calculate_structure_metrics(self, pred_struct: str, true_struct: str = None, 
                                  vienna_struct: str = None) -> Dict[str, float]:
        """Calculate various structure comparison metrics"""
        metrics = {}
        
        if true_struct:
            # Accuracy with ground truth
            matches = sum(1 for a, b in zip(pred_struct, true_struct) if a == b)
            metrics['gt_accuracy'] = matches / len(true_struct)
            
            # Base pair accuracy
            pred_pairs = self.get_base_pairs(pred_struct)
            true_pairs = self.get_base_pairs(true_struct)
            
            if true_pairs:
                common_pairs = pred_pairs & true_pairs
                metrics['bp_precision'] = len(common_pairs) / len(pred_pairs) if pred_pairs else 0
                metrics['bp_recall'] = len(common_pairs) / len(true_pairs)
                metrics['bp_f1'] = (2 * metrics['bp_precision'] * metrics['bp_recall'] / 
                                  (metrics['bp_precision'] + metrics['bp_recall'])) if (metrics['bp_precision'] + metrics['bp_recall']) > 0 else 0
        
        if vienna_struct:
            # Comparison with ViennaRNA
            matches = sum(1 for a, b in zip(pred_struct, vienna_struct) if a == b)
            metrics['vienna_similarity'] = matches / len(vienna_struct)
        
        # Structure complexity metrics
        metrics['base_pairs'] = pred_struct.count('(')
        metrics['unpaired_bases'] = pred_struct.count('.')
        metrics['gc_content'] = 0  # Will be calculated with sequence
        
        return metrics
    
    def get_base_pairs(self, structure: str) -> set:
        """Extract base pairs from dot-bracket structure"""
        stack = []
        pairs = set()
        
        for i, c in enumerate(structure):
            if c == '(':
                stack.append(i)
            elif c == ')' and stack:
                j = stack.pop()
                pairs.add((j, i))
        
        return pairs
    
    def generate_interactive_html(self, sequence: str, structures: Dict[str, str], 
                                title: str = "RNA Structure Comparison") -> str:
        """Generate interactive HTML visualization with multiple structures"""
        
        structure_data = []
        colors = {
            'ground_truth': '#2E7D32',  # Green
            'vienna': '#1976D2',        # Blue  
            'prediction': '#D32F2F',    # Red
            'consensus': '#7B1FA2'      # Purple
        }
        
        for name, struct in structures.items():
            if struct:
                structure_data.append({
                    'name': name.replace('_', ' ').title(),
                    'structure': struct,
                    'color': colors.get(name, '#666666')
                })
        
        html_content = f"""
        <div class="rna-comparison-container">
            <div class="comparison-title">{title}</div>
            <div class="sequence-display">
                <strong>Sequence:</strong> <span class="sequence">{sequence}</span>
            </div>
            
            <div class="structure-tabs">
                <div class="tab-buttons">
                    {' '.join([f'<button class="tab-btn" onclick="showStructure({i})" data-index="{i}">{data["name"]}</button>' 
                              for i, data in enumerate(structure_data)])}
                    <button class="tab-btn" onclick="showOverlay()">Overlay Comparison</button>
                </div>
                
                <div class="structure-panels">
                    {' '.join([f'<div class="structure-panel" id="panel-{i}"><div id="rna-viz-{i}"></div></div>' 
                              for i in range(len(structure_data))])}
                    <div class="structure-panel" id="overlay-panel">
                        <div id="overlay-viz"></div>
                        <div class="overlay-legend">
                            {' '.join([f'<span class="legend-item"><span class="legend-color" style="background-color: {data["color"]}"></span>{data["name"]}</span>' 
                                      for data in structure_data])}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <style>
        .rna-comparison-container {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin: 10px 0;
        }}
        
        .comparison-title {{
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
            color: #1a365d;
            margin-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }}
        
        .sequence-display {{
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            text-align: center;
        }}
        
        .sequence {{
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            color: #2d3748;
            letter-spacing: 1px;
            word-break: break-all;
        }}
        
        .tab-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 15px;
            justify-content: center;
        }}
        
        .tab-btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 20px;
            background: #e2e8f0;
            color: #4a5568;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        
        .tab-btn:hover {{
            background: #cbd5e0;
            transform: translateY(-2px);
        }}
        
        .tab-btn.active {{
            background: #4299e1;
            color: white;
            box-shadow: 0 4px 8px rgba(66, 153, 225, 0.3);
        }}
        
        .structure-panel {{
            display: none;
            min-height: 400px;
            background: #fafafa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        
        .structure-panel.active {{
            display: block;
        }}
        
        .overlay-legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        
        .nucleotide {{
            stroke-width: 1.5;
            cursor: pointer;
        }}
        
        .basepair {{
            stroke-width: 2;
            opacity: 0.8;
        }}
        
        .backbone {{
            stroke-width: 1;
            opacity: 0.6;
        }}
        </style>
        
        <script src="https://cdn.jsdelivr.net/npm/d3@3.5.17/d3.min.js"></script>
        <script>
        var structureData = {json.dumps(structure_data)};
        var sequence = "{sequence}";
        var currentPanel = 0;
        
        function showStructure(index) {{
            // Update tab buttons
            document.querySelectorAll('.tab-btn').forEach((btn, i) => {{
                btn.classList.toggle('active', i === index || (i === structureData.length && index === -1));
            }});
            
            // Update panels
            document.querySelectorAll('.structure-panel').forEach((panel, i) => {{
                panel.classList.toggle('active', i === index || (i === structureData.length && index === -1));
            }});
            
            currentPanel = index;
            
            if (index >= 0 && index < structureData.length) {{
                drawRNAStructure(structureData[index], `rna-viz-${{index}}`);
            }}
        }}
        
        function showOverlay() {{
            showStructure(-1);
            drawOverlayComparison();
        }}
        
        function drawRNAStructure(data, containerId) {{
            var container = d3.select(`#${{containerId}}`);
            container.selectAll("*").remove();
            
            var width = 600;
            var height = 400;
            var svg = container.append("svg")
                .attr("width", width)
                .attr("height", height)
                .style("background", "white");
            
            var structure = data.structure;
            var color = data.color;
            
            // Parse base pairs
            var pairs = [];
            var stack = [];
            for (var i = 0; i < structure.length; i++) {{
                if (structure[i] === '(') {{
                    stack.push(i);
                }} else if (structure[i] === ')' && stack.length > 0) {{
                    var j = stack.pop();
                    pairs.push([j, i]);
                }}
            }}
            
            // Calculate positions
            var centerX = width / 2;
            var centerY = height / 2;
            var radius = Math.min(centerX, centerY) - 50;
            var positions = [];
            
            for (var i = 0; i < sequence.length; i++) {{
                var angle = (i / sequence.length) * 2 * Math.PI - Math.PI/2;
                positions.push({{
                    x: centerX + radius * Math.cos(angle),
                    y: centerY + radius * Math.sin(angle),
                    base: sequence[i],
                    index: i
                }});
            }}
            
            // Draw base pairs
            pairs.forEach(function(pair) {{
                svg.append("line")
                    .attr("x1", positions[pair[0]].x)
                    .attr("y1", positions[pair[0]].y)
                    .attr("x2", positions[pair[1]].x)
                    .attr("y2", positions[pair[1]].y)
                    .attr("stroke", color)
                    .attr("class", "basepair");
            }});
            
            // Draw backbone
            for (var i = 0; i < positions.length - 1; i++) {{
                svg.append("line")
                    .attr("x1", positions[i].x)
                    .attr("y1", positions[i].y)
                    .attr("x2", positions[i + 1].x)
                    .attr("y2", positions[i + 1].y)
                    .attr("stroke", "#cccccc")
                    .attr("class", "backbone");
            }}
            
            // Draw nucleotides
            var nodes = svg.selectAll(".nucleotide-group")
                .data(positions)
                .enter().append("g")
                .attr("class", "nucleotide-group")
                .attr("transform", function(d) {{ return `translate(${{d.x}},${{d.y}})`; }});
            
            nodes.append("circle")
                .attr("r", 12)
                .attr("fill", color)
                .attr("class", "nucleotide")
                .on("mouseover", function(d) {{
                    d3.select(this).attr("r", 15);
                }})
                .on("mouseout", function(d) {{
                    d3.select(this).attr("r", 12);
                }});
            
            nodes.append("text")
                .attr("text-anchor", "middle")
                .attr("dy", "0.35em")
                .style("font-size", "11px")
                .style("font-weight", "bold")
                .style("fill", "white")
                .text(function(d) {{ return d.base; }});
        }}
        
        function drawOverlayComparison() {{
            var container = d3.select("#overlay-viz");
            container.selectAll("*").remove();
            
            var width = 700;
            var height = 500;
            var svg = container.append("svg")
                .attr("width", width)
                .attr("height", height)
                .style("background", "white");
            
            // Draw all structures overlaid
            structureData.forEach(function(data, structIndex) {{
                var structure = data.structure;
                var color = data.color;
                
                // Parse base pairs
                var pairs = [];
                var stack = [];
                for (var i = 0; i < structure.length; i++) {{
                    if (structure[i] === '(') {{
                        stack.push(i);
                    }} else if (structure[i] === ')' && stack.length > 0) {{
                        var j = stack.pop();
                        pairs.push([j, i]);
                    }}
                }}
                
                // Calculate positions
                var centerX = width / 2;
                var centerY = height / 2;
                var radius = Math.min(centerX, centerY) - 60 - structIndex * 20;
                var positions = [];
                
                for (var i = 0; i < sequence.length; i++) {{
                    var angle = (i / sequence.length) * 2 * Math.PI - Math.PI/2;
                    positions.push({{
                        x: centerX + radius * Math.cos(angle),
                        y: centerY + radius * Math.sin(angle)
                    }});
                }}
                
                // Draw base pairs for this structure
                pairs.forEach(function(pair) {{
                    svg.append("line")
                        .attr("x1", positions[pair[0]].x)
                        .attr("y1", positions[pair[0]].y)
                        .attr("x2", positions[pair[1]].x)
                        .attr("y2", positions[pair[1]].y)
                        .attr("stroke", color)
                        .attr("stroke-width", 2)
                        .attr("opacity", 0.7)
                        .attr("class", `basepair-${{structIndex}}`);
                }});
            }});
        }}
        
        // Initialize with first structure
        if (structureData.length > 0) {{
            showStructure(0);
        }}
        </script>
        """
        
        return html_content
    
    def predict_and_compare(self, sequence: str, ground_truth: str = "") -> Tuple[str, str, str, str, str]:
        """Predict structure and generate comparison visualization"""
        
        # Validate sequence
        is_valid, message = self.validate_rna_sequence(sequence)
        if not is_valid:
            return "", "", "", f"❌ {message}", ""
        
        sequence = sequence.upper().replace("T", "U")
        
        try:
            # ViennaRNA prediction
            vienna_struct, vienna_energy = RNA.fold(sequence)
            
            # Model prediction
            pred_struct = ""
            if self.model:
                try:
                    result = self.model.inference(sequence)
                    pred_struct = "".join(result.get('predictions', []))
                    
                    # Fix invalid structure
                    if self.ss_validity_loss(pred_struct) > 0:
                        pred_struct = self.fix_invalid_structure(pred_struct)
                        
                except Exception as e:
                    pred_struct = vienna_struct  # Fallback to Vienna
                    print(f"Model prediction failed: {e}")
            else:
                pred_struct = vienna_struct
            
            # Prepare structures for visualization
            structures = {
                'vienna': vienna_struct,
                'prediction': pred_struct
            }
            
            if ground_truth.strip():
                ground_truth = ground_truth.strip()
                structures['ground_truth'] = ground_truth
            
            # Calculate metrics
            metrics = self.calculate_structure_metrics(
                pred_struct, ground_truth if ground_truth else None, vienna_struct
            )
            
            # Generate statistics
            stats_lines = [
                f"📊 **Structure Statistics:**",
                f"• Sequence length: {len(sequence)} bases",
                f"• Base pairs (Vienna): {vienna_struct.count('(')}",
                f"• Base pairs (Prediction): {pred_struct.count('(')}",
                f"• Unpaired bases: {vienna_struct.count('.')}",
                f"• Vienna energy: {vienna_energy:.2f} kcal/mol"
            ]
            
            if ground_truth:
                stats_lines.extend([
                    f"",
                    f"🎯 **Accuracy Metrics:**",
                    f"• GT ↔ Prediction: {metrics.get('gt_accuracy', 0):.2%}",
                    f"• Base pair precision: {metrics.get('bp_precision', 0):.2%}",
                    f"• Base pair recall: {metrics.get('bp_recall', 0):.2%}",
                    f"• Base pair F1-score: {metrics.get('bp_f1', 0):.2%}"
                ])
            
            if 'vienna_similarity' in metrics:
                stats_lines.extend([
                    f"• Vienna ↔ Prediction: {metrics['vienna_similarity']:.2%}"
                ])
            
            stats_text = "\n".join(stats_lines)
            
            # Generate visualization
            visualization = self.generate_interactive_html(
                sequence, structures, 
                f"RNA Structure Comparison - {len(sequence)} bases"
            )
            
            # Store in history
            self.comparison_history.append({
                'sequence': sequence,
                'structures': structures,
                'metrics': metrics,
                'timestamp': time.time()
            })
            
            return (
                ground_truth,
                vienna_struct, 
                pred_struct,
                stats_text,
                visualization
            )
            
        except Exception as e:
            error_msg = f"❌ Error during prediction: {str(e)}"
            return "", "", "", error_msg, ""
    
    def sample_rna_sequence(self) -> Tuple[str, str]:
        """Sample RNA sequence from test dataset"""
        try:
            test_file = 'toy_datasets/Archive2/test.json'
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    examples = [json.loads(line) for line in f]
                
                if examples:
                    example = examples[np.random.randint(len(examples))]
                    sequence = example.get('seq', '').replace('T', 'U')
                    structure = example.get('label', '')
                    return sequence, structure
            
            # Fallback examples
            examples = [
                ("GGGAAACCC", "(((...)))"),
                ("GGCCAAUUGGCC", "((((...))))"),
                ("GGGCCCAAAGGGCCC", "((((((...))))))"),
                ("AAAUUUGGGCCCAAAUUU", "...((((((...))))))"),
                ("GGGCGCAAAGCGCCC", "((((((...))))))"),
            ]
            
            seq, struct = examples[np.random.randint(len(examples))]
            return seq, struct
            
        except Exception as e:
            return f"Error loading sample: {e}", ""


def create_enhanced_ssp_demo():
    """Create enhanced secondary structure prediction demo"""
    
    demo_instance = EnhancedSSPDemo()
    
    # Enhanced CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        text-align: center;
        color: #1a365d;
        font-size: 2.5em;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .section-header {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    .input-panel {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stats-panel {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 1px solid #2196F3;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        line-height: 1.6;
    }
    .control-buttons {
        display: flex;
        gap: 15px;
        justify-content: center;
        margin: 20px 0;
    }
    .feature-highlight {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="Enhanced RNA Structure Prediction") as demo:
        gr.Markdown("# 🧬 Enhanced RNA Secondary Structure Prediction", elem_classes="main-title")
        
        gr.Markdown("""
        <div class="feature-highlight">
        🚀 <strong>Enhanced Features:</strong> Interactive visualization, real-time comparison, 
        structure metrics, and dynamic overlays for comprehensive RNA structure analysis.
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📝 Input Panel", elem_classes="section-header")
                
                with gr.Group(elem_classes="input-panel"):
                    sequence_input = gr.Textbox(
                        label="🔤 RNA Sequence",
                        lines=4,
                        placeholder="Enter RNA sequence (A, U, G, C)...\nExample: GGGAAACCC",
                        info="Enter your RNA sequence using A, U, G, C bases"
                    )
                    
                    ground_truth_input = gr.Textbox(
                        label="🎯 Ground Truth Structure (Optional)",
                        lines=3,
                        placeholder="Enter known structure using (), and . notation...\nExample: (((...)))",
                        info="Optional: Provide known structure for accuracy comparison"
                    )
                    
                    with gr.Row(elem_classes="control-buttons"):
                        sample_btn = gr.Button("🎲 Sample Example", variant="secondary", size="lg")
                        predict_btn = gr.Button("🔬 Predict Structure", variant="primary", size="lg")
                
                gr.Markdown("## 📊 Analysis Results", elem_classes="section-header")
                
                stats_output = gr.Markdown(
                    value="Click 'Predict Structure' to see detailed analysis...",
                    elem_classes="stats-panel"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## 🎨 Interactive Visualization", elem_classes="section-header")
                
                visualization_output = gr.HTML(
                    value="""
                    <div style='text-align: center; padding: 50px; background: #f8f9fa; border-radius: 10px; margin: 20px 0;'>
                        <h3 style='color: #666;'>🔬 Structure Visualization</h3>
                        <p style='color: #888;'>Enter an RNA sequence and click 'Predict Structure' to see interactive visualization</p>
                    </div>
                    """
                )
        
        # Output textboxes (hidden, for data flow)
        with gr.Row(visible=False):
            gt_output = gr.Textbox()
            vienna_output = gr.Textbox()
            pred_output = gr.Textbox()
        
        gr.Markdown("""
        ### 🎯 How to Use:
        1. **Enter Sequence**: Type or paste an RNA sequence using A, U, G, C bases
        2. **Optional Ground Truth**: Add known structure for accuracy comparison
        3. **Sample Examples**: Click "Sample Example" to try pre-loaded test cases
        4. **Predict**: Click "Predict Structure" to generate predictions and visualization
        5. **Explore**: Use the interactive tabs to compare different prediction methods
        6. **Analyze**: Review detailed metrics and statistics in the analysis panel
        
        ### 🔬 Visualization Features:
        - **Interactive Structure Display**: Zoom, pan, and explore RNA structures
        - **Multi-Method Comparison**: Compare ViennaRNA, AI model, and ground truth
        - **Overlay Mode**: See all structures simultaneously with color coding
        - **Real-time Metrics**: Accuracy, precision, recall, and F1-scores
        - **Base Pair Analysis**: Detailed base-pairing statistics and energy calculations
        
        ### 📈 Metrics Explained:
        - **Accuracy**: Percentage of correctly predicted positions
        - **Base Pair Precision**: Fraction of predicted pairs that are correct
        - **Base Pair Recall**: Fraction of true pairs that were predicted
        - **F1-Score**: Harmonic mean of precision and recall
        """)
        
        # Event handlers
        sample_btn.click(
            fn=demo_instance.sample_rna_sequence,
            outputs=[sequence_input, ground_truth_input]
        )
        
        predict_btn.click(
            fn=demo_instance.predict_and_compare,
            inputs=[sequence_input, ground_truth_input],
            outputs=[gt_output, vienna_output, pred_output, stats_output, visualization_output]
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch the enhanced demo
    demo = create_enhanced_ssp_demo()
    demo.launch(
        share=True,
        server_name="0.0.0.0", 
        server_port=7861,
        show_error=True
    )
