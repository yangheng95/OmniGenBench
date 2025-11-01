.. _troubleshooting:

#####################
Troubleshooting Guide
#####################

This comprehensive guide addresses common issues encountered when using OmniGenBench, organized by category. For each problem, we provide symptoms, root cause analysis, and systematic solutions with working code examples.

**Quick Navigation:**

* :ref:`installation-issues` - ViennaRNA, PyTorch, dependency conflicts
* :ref:`cli-issues` - Command-line interface errors and usage patterns
* :ref:`runtime-errors` - Out of memory, sequence length, ViennaRNA fold errors
* :ref:`model-loading-issues` - Tokenizer, architecture mismatch, trust_remote_code
* :ref:`training-evaluation-issues` - NaN loss, poor performance, evaluation metrics
* :ref:`platform-specific` - Windows encoding, path issues, long path support

.. _cli-issues:

***********************
Command-Line Interface (CLI) Issues
***********************

**Problem: Command Not Found - 'ogb' or 'autoinfer'**

**Symptoms:**

.. code-block:: text

   bash: ogb: command not found
   bash: autoinfer: command not found

**Solutions:**

1. **Verify Installation**:

   .. code-block:: bash
   
      pip show omnigenbench
      # Check that 'Location' points to active Python environment

2. **Check PATH Configuration**:

   .. code-block:: bash
   
      # Verify pip scripts directory is in PATH
      which ogb  # Linux/Mac
      where ogb  # Windows
      
      # If not found, add to PATH
      export PATH="$HOME/.local/bin:$PATH"  # Linux
      export PATH="$(python -m site --user-base)/bin:$PATH"  # macOS

3. **Reinstall in Correct Environment**:

   .. code-block:: bash
   
      conda activate omnigen_env
      pip install --force-reinstall omnigenbench

4. **Use Python Module Execution** (Fallback):

   .. code-block:: bash
   
      python -m omnigenbench.cli.ogb_cli autobench --help

**Problem: Invalid Argument Errors**

**Symptoms:**

.. code-block:: text

   error: unrecognized arguments: --seeds 0 1 2
   error: argument --model: expected one argument

**Solutions:**

1. **Check Argument Syntax** (Note: ``--seeds`` takes multiple values):

   .. code-block:: bash
   
      # CORRECT: Multiple seeds without commas
      ogb autobench --model model --benchmark RGB --seeds 0 1 2
      
      # WRONG: Comma-separated or quoted
      # ogb autobench --model model --benchmark RGB --seeds "0,1,2"

2. **Verify Required Arguments**:

   .. code-block:: bash
   
      # View all required and optional arguments
      ogb autobench --help
      ogb autotrain --help
      ogb autoinfer --help
      ogb rna_design --help

3. **Common Argument Patterns**:

   .. code-block:: bash
   
      # AutoBench: model and benchmark are required
      ogb autobench --model yangheng/OmniGenome-186M --benchmark RGB
      
      # AutoTrain: dataset and model are required
      ogb autotrain --dataset ./data --model yangheng/OmniGenome-186M
      
      # AutoInfer: model and (sequence OR input-file) required
      ogb autoinfer --model yangheng/ogb_tfb_finetuned --sequence "ATCG"
      ogb autoinfer --model yangheng/ogb_tfb_finetuned --input-file data.json
      
      # RNA Design: structure is required
      ogb rna_design --structure "(((...)))"

**Problem: Benchmark Name Not Recognized**

**Symptoms:**

.. code-block:: text

   ValueError: Benchmark 'rgb' not found. Available: RGB, BEACON, PGB, GUE, GB

**Solutions:**

1. **Use Correct Case-Sensitive Names**:

   .. code-block:: bash
   
      # CORRECT: Uppercase benchmark names
      ogb autobench --model model --benchmark RGB
      ogb autobench --model model --benchmark BEACON
      ogb autobench --model model --benchmark PGB
      ogb autobench --model model --benchmark GUE
      ogb autobench --model model --benchmark GB
      
      # WRONG: Lowercase will fail
      # ogb autobench --model model --benchmark rgb

2. **List Available Benchmarks**:

   .. code-block:: python
   
      from omnigenbench import BenchHub
      print(BenchHub.list_benchmarks())

**Problem: Trainer Backend Not Available**

**Symptoms:**

.. code-block:: text

   ValueError: Trainer 'Accelerate' not recognized. Use 'native', 'accelerate', or 'hf_trainer'

**Solutions:**

1. **Use Lowercase Trainer Names**:

   .. code-block:: bash
   
      # CORRECT: Lowercase trainer names
      ogb autobench --model model --benchmark RGB --trainer native
      ogb autobench --model model --benchmark RGB --trainer accelerate
      ogb autobench --model model --benchmark RGB --trainer hf_trainer
      
      # WRONG: Capitalized will fail
      # ogb autobench --model model --benchmark RGB --trainer Accelerate

2. **Install Missing Dependencies**:

   .. code-block:: bash
   
      # For accelerate trainer
      pip install accelerate
      
      # For hf_trainer
      pip install transformers[torch]

3. **Verify Trainer Availability**:

   .. code-block:: python
   
      try:
          from accelerate import Accelerator
          print("Accelerate trainer available")
      except ImportError:
          print("Install: pip install accelerate")

.. _installation-issues:

***********************
Installation Issues
***********************

**Problem: ViennaRNA Installation Fails**

**Symptoms:**

.. code-block:: text

   ERROR: Could not build wheels for viennarna
   error: command 'gcc' failed with exit status 1

**Solutions:**

1. **Use Conda (Recommended)**:

   .. code-block:: bash
   
      conda install -c bioconda viennarna
      pip install omnigenbench

2. **Install Build Dependencies** (Linux):

   .. code-block:: bash
   
      sudo apt-get update
      sudo apt-get install build-essential python3-dev
      pip install viennarna

3. **Use Pre-built Binary** (macOS):

   .. code-block:: bash
   
      brew install viennarna
      pip install viennarna

4. **Windows Users**: Use WSL2 or Docker:

   .. code-block:: bash
   
      # Inside WSL2
      sudo apt-get install python3-viennarna

**Problem: PyTorch CUDA Mismatch**

**Symptoms:**

.. code-block:: python

   RuntimeError: CUDA error: no kernel image is available for execution on the device
   torch.cuda.is_available() returns False

**Solutions:**

1. **Check CUDA Version**:

   .. code-block:: bash
   
      nvidia-smi  # Look at CUDA Version in top right
      python -c "import torch; print(torch.version.cuda)"

2. **Reinstall Matching PyTorch**:

   .. code-block:: bash
   
      # For CUDA 11.8
      pip uninstall torch
      pip install torch --index-url https://download.pytorch.org/whl/cu118
      
      # For CUDA 12.1
      pip install torch --index-url https://download.pytorch.org/whl/cu121

3. **Verify Installation**:

   .. code-block:: python
   
      import torch
      print(f"CUDA available: {torch.cuda.is_available()}")
      print(f"CUDA version: {torch.version.cuda}")
      print(f"Device count: {torch.cuda.device_count()}")

**Problem: Import Errors After Installation**

**Symptoms:**

.. code-block:: text

   ModuleNotFoundError: No module named 'omnigenbench'
   ImportError: cannot import name 'OmniModelForSequenceClassification'

**Solutions:**

1. **Verify Installation**:

   .. code-block:: bash
   
      pip show omnigenbench
      # Check that package is installed and version is correct

2. **Check Python Environment**:

   .. code-block:: bash
   
      # Ensure you're in the correct environment
      which python  # Linux/Mac
      where python  # Windows
      
      # Reinstall in correct environment
      pip install --upgrade omnigenbench

3. **Development Installation** (if working from source):

   .. code-block:: bash
   
      cd /path/to/OmniGenBench
      pip install -e .  # Editable install

4. **Clear Python Cache**:

   .. code-block:: bash
   
      # Remove cached bytecode
      find . -type d -name "__pycache__" -exec rm -rf {} +  # Linux/Mac
      # Windows: Delete __pycache__ folders manually

.. _platform-specific:

**Problem: Windows Emoji/Unicode Display Issues**

**Symptoms:**

.. code-block:: text

   UnicodeEncodeError: 'charmap' codec can't encode character
   Display shows garbled characters instead of progress bars

**Solutions:**

1. **Set UTF-8 Encoding** (Recommended):

   .. code-block:: bash
   
      # PowerShell
      $env:PYTHONIOENCODING="utf-8"
      [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
      
      # Git Bash (recommended for Windows)
      export PYTHONIOENCODING=utf-8

2. **Use Git Bash Instead of CMD**:

   Git Bash provides better Unicode support and is the recommended terminal for Windows users.

3. **Disable Emoji Output** (if issues persist):

   OmniGenBench avoids emoji in output by design for Windows compatibility. If you see
   encoding errors, they likely come from custom print statements or third-party libraries.

**Problem: Windows Path Issues**

**Symptoms:**

.. code-block:: text

   FileNotFoundError: [Errno 2] No such file or directory: 'results\\model'
   OSError: [WinError 123] Invalid filename, directory name, or volume label

**Solutions:**

1. **Use Forward Slashes**:

   .. code-block:: python
   
      # Good - works on all platforms
      model.save_model("results/my_model")
      dataset = OmniDataset("data/sequences.json", tokenizer)
      
      # Avoid - Windows-specific backslashes
      # model.save_model("results\\my_model")

2. **Use pathlib for Cross-Platform Compatibility**:

   .. code-block:: python
   
      from pathlib import Path
      
      output_dir = Path("results") / "models" / "experiment_1"
      output_dir.mkdir(parents=True, exist_ok=True)
      model.save_model(str(output_dir))

3. **Avoid Long Paths** (Windows 260-character limit):

   .. code-block:: bash
   
      # Enable long path support (Windows 10+, requires admin)
      # Run in PowerShell as Administrator:
      New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                       -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

**Solutions:**

1. **Verify Installation**:

   .. code-block:: bash
   
      pip show omnigenbench
      which python  # Ensure correct environment

2. **Check Virtual Environment**:

   .. code-block:: bash
   
      conda activate omnigen_env
      pip list | grep omnigenbench

3. **Reinstall in Clean Environment**:

   .. code-block:: bash
   
      conda create -n omnigen_fresh python=3.12
      conda activate omnigen_fresh
      pip install omnigenbench

.. _runtime-errors:

***********************
Runtime Errors
***********************

**Problem: Out of Memory (OOM) Errors**

**Symptoms:**

.. code-block:: text

   RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
   torch.cuda.OutOfMemoryError

**Solutions:**

1. **Reduce Batch Size**:

   .. code-block:: python
   
      # Python API
      bench.run(batch_size=8)  # Instead of 32
      
      # CLI
      ogb autobench --model model --benchmark RGB --batch-size 8

2. **Enable Gradient Checkpointing** (for training):

   .. code-block:: python
   
      from omnigenbench import AutoTrain
      
      trainer = AutoTrain(
          dataset="data",
          config_or_model="model",
          gradient_checkpointing=True  # Trades compute for memory
      )

3. **Use Mixed Precision**:

   .. code-block:: bash
   
      ogb autotrain --dataset data --model model --autocast

4. **Monitor GPU Memory**:

   .. code-block:: bash
   
      watch -n 1 nvidia-smi
      # Or
      nvidia-smi dmon -s u -d 1

5. **Clear GPU Cache**:

   .. code-block:: python
   
      import torch
      torch.cuda.empty_cache()

**Problem: Sequence Length Exceeds Max Length**

**Symptoms:**

.. code-block:: text

   ValueError: Input length 8192 exceeds maximum length 512
   RuntimeError: The size of tensor a (8192) must match the size of tensor b (512)

**Solutions:**

1. **Increase max_length Parameter**:

   .. code-block:: python
   
      dataset = OmniDatasetForSequenceClassification(
          dataset_name_or_path="data.json",
          tokenizer=tokenizer,
          max_length=8192  # Increase from default 512
      )
      
      # For models, ensure they support longer sequences
      model = OmniModelForSequenceClassification(
          "yangheng/OmniGenome-186M",
          tokenizer=tokenizer,
          num_labels=2
      )
      # Note: Base model must support the target sequence length

2. **Use drop_long_seq to Filter**:

   .. code-block:: python
   
      dataset = OmniDatasetForSequenceClassification(
          dataset_name_or_path="data.json",
          tokenizer=tokenizer,
          max_length=512,
          drop_long_seq=True  # Drop sequences > max_length instead of truncating
      )

3. **Chunking Long Sequences** (for very long genomic regions):

   .. code-block:: python
   
      def chunk_sequence(seq, chunk_size=512, overlap=50):
          """Split long sequence into overlapping chunks."""
          chunks = []
          for i in range(0, len(seq), chunk_size - overlap):
              chunk = seq[i:i + chunk_size]
              if len(chunk) >= chunk_size // 2:  # Keep meaningful chunks
                  chunks.append(chunk)
          return chunks
      
      # Process each chunk separately
      long_seq = "ATCG" * 3000  # 12000 bp sequence
      chunks = chunk_sequence(long_seq, chunk_size=512)
      results = [model.inference(chunk) for chunk in chunks]

**Problem: ViennaRNA Fold Function Errors**

**Symptoms:**

.. code-block:: text

   AttributeError: module 'RNA' has no attribute 'fold'
   ImportError: No module named 'RNA'

**Solutions:**

1. **Install ViennaRNA** (Required for RNA structure prediction/design):

   .. code-block:: bash
   
      # Conda (recommended)
      conda install -c bioconda viennarna
      
      # Linux
      sudo apt-get install python3-viennarna
      
      # macOS
      brew install viennarna

2. **Verify Installation**:

   .. code-block:: python
   
      try:
          import RNA
          structure, mfe = RNA.fold("GCGAAACGC")
          print(f"Structure: {structure}, MFE: {mfe}")
      except ImportError:
          print("ViennaRNA not installed")

3. **Windows Users**: ViennaRNA has limited Windows support. Options:
   
   - Use WSL2 (Windows Subsystem for Linux)
   - Use Docker container with ViennaRNA
   - Use online RNA folding services as fallback

   .. code-block:: python
   
      dataset = OmniDatasetForSequenceClassification(
          data_path="data.json",
          tokenizer=tokenizer,
          max_length=8192  # Increase from default 512
      )

2. **Use Model with Longer Context**:

   .. code-block:: python
   
      # Models with long-context support
      model = ModelHub.load("LongSafari/hyenadna-medium-160k-seqlen-hf")  # 160k tokens

3. **Truncate Sequences**:

   .. code-block:: python
   
      dataset = OmniDatasetForSequenceClassification(
          data_path="data.json",
          tokenizer=tokenizer,
          max_length=512,
          truncation=True  # Enable truncation
      )

**Problem: HuggingFace Hub Authentication Errors**

**Symptoms:**

.. code-block:: text

   HTTPError: 401 Client Error: Unauthorized
   Repository not found

**Solutions:**

1. **Login to HuggingFace**:

   .. code-block:: bash
   
      huggingface-cli login
      # Enter your token from https://huggingface.co/settings/tokens

2. **Set Environment Variable**:

   .. code-block:: bash
   
      export HUGGINGFACE_TOKEN=hf_your_token_here

3. **Use Access Token in Code**:

   .. code-block:: python
   
      from huggingface_hub import login
      login(token="hf_your_token_here")

**Problem: Windows Encoding Errors**

**Symptoms:**

.. code-block:: text

   UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'

**Solutions:**

1. **Set Terminal Encoding** (PowerShell):

   .. code-block:: powershell
   
      [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

2. **Use Git Bash** (Recommended for Windows):

   .. code-block:: bash
   
      export PYTHONIOENCODING=utf-8

3. **Disable Unicode in Output**:

   .. code-block:: bash
   
      ogb autobench --model model --benchmark RGB --no-unicode

.. _model-loading-issues:

***********************
Model Loading Issues
***********************

**Problem: Tokenizer Not Found**

**Symptoms:**

.. code-block:: text

   OSError: Can't load tokenizer for 'yangheng/OmniGenome-186M'

**Solutions:**

1. **Verify Model Exists**:

   .. code-block:: bash
   
      # Check on HuggingFace Hub
      # https://huggingface.co/yangheng/OmniGenome-186M

2. **Specify Tokenizer Explicitly**:

   .. code-block:: python
   
      from omnigenbench import ModelHub, OmniSingleNucleotideTokenizer
      
      tokenizer = OmniSingleNucleotideTokenizer.from_pretrained("model")
      model = ModelHub.load("model", tokenizer=tokenizer)

3. **Use Local Tokenizer**:

   .. code-block:: python
   
      tokenizer = OmniSingleNucleotideTokenizer(
          vocab_file="./tokenizer_vocab.json"
      )

**Problem: Model Architecture Mismatch**

**Symptoms:**

.. code-block:: text

   RuntimeError: Error(s) in loading state_dict for BertModel:
   size mismatch for embeddings.word_embeddings.weight

**Solutions:**

1. **Use Correct Task-Specific Model Class**:

   .. code-block:: python
   
      # WRONG: Generic model class
      from omnigenbench import OmniModel
      
      # RIGHT: Task-specific model class
      from omnigenbench import OmniModelForSequenceClassification
      
      model = OmniModelForSequenceClassification(
          config_or_model="yangheng/ogb_tfb_finetuned",
          num_labels=919  # Match training configuration
      )

2. **Check Model Configuration**:

   .. code-block:: python
   
      from transformers import AutoConfig
      
      config = AutoConfig.from_pretrained("model")
      print(config)  # Verify num_labels, hidden_size, etc.

**Problem: trust_remote_code Error**

**Symptoms:**

.. code-block:: text

   ValueError: Loading this model requires you to execute code in the model
   repository. You can enable this by setting `trust_remote_code=True`

**Solutions:**

1. **Enable trust_remote_code**:

   .. code-block:: python
   
      model = ModelHub.load(
          "yangheng/OmniGenome-186M",
          trust_remote_code=True
      )

2. **Understand Security Implications**: Only use for trusted models from reputable sources.

.. _training-evaluation-issues:

***********************
Training & Evaluation Issues
***********************

**Problem: Training Loss is NaN**

**Symptoms:**

.. code-block:: text

   Epoch 1: loss = nan
   RuntimeError: loss is nan

**Solutions:**

1. **Reduce Learning Rate**:

   .. code-block:: python
   
      trainer = AutoTrain(
          dataset="data",
          config_or_model="model",
          learning_rate=1e-5  # Instead of default 2e-5
      )

2. **Enable Gradient Clipping**:

   .. code-block:: python
   
      trainer = AutoTrain(
          dataset="data",
          config_or_model="model",
          max_grad_norm=1.0  # Clip gradients
      )

3. **Check Data Quality**:

   .. code-block:: python
   
      # Verify no NaN or Inf in labels
      import json
      data = json.load(open("train.json"))
      labels = [d['label'] for d in data]
      print(f"NaN count: {sum([l != l for l in labels])}")  # l != l checks for NaN

4. **Use Mixed Precision Carefully**:

   .. code-block:: bash
   
      # Try without autocast first
      ogb autotrain --dataset data --model model

**Problem: Poor Model Performance**

**Symptoms:**

.. code-block:: text

   Test accuracy: 0.52 (close to random)
   MCC: 0.05

**Solutions:**

1. **Increase Training Epochs**:

   .. code-block:: bash
   
      ogb autotrain --dataset data --model model --num-epochs 100

2. **Adjust Learning Rate**:

   .. code-block:: python
   
      # Try learning rate sweep
      for lr in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]:
          trainer = AutoTrain(
              dataset="data",
              config_or_model="model",
              learning_rate=lr
          )
          trainer.run()

3. **Use Multi-Seed Evaluation**:

   .. code-block:: python
   
      bench = AutoBench(
          benchmark="RGB",
          config_or_model="model"
      )
      bench.run(seeds=[0, 1, 2, 3, 4])  # Average over 5 runs

4. **Verify Data Quality**:

   .. code-block:: python
   
      # Check class balance
      import json
      from collections import Counter
      
      data = json.load(open("train.json"))
      labels = [d['label'] for d in data]
      print(Counter(labels))  # Should not be extremely imbalanced

5. **Try Different Model**:

   .. code-block:: bash
   
      # Models with different architectures
      ogb autotrain --dataset data --model zhihan1996/DNABERT-2-117M
      ogb autotrain --dataset data --model yangheng/OmniGenome-186M

**Problem: Slow Training Speed**

**Symptoms:**

.. code-block:: text

   Training speed: 0.5 it/s (expected 5-10 it/s)

**Solutions:**

1. **Use Multi-GPU Training**:

   .. code-block:: bash
   
      ogb autotrain --dataset data --model model --trainer accelerate

2. **Increase Batch Size**:

   .. code-block:: bash
   
      ogb autotrain --dataset data --model model --batch-size 64

3. **Enable Mixed Precision**:

   .. code-block:: bash
   
      ogb autotrain --dataset data --model model --autocast

4. **Use DataLoader Workers**:

   .. code-block:: python
   
      trainer = AutoTrain(
          dataset="data",
          config_or_model="model",
          num_workers=4  # Parallel data loading
      )

5. **Profile Bottlenecks**:

   .. code-block:: python
   
      import time
      
      start = time.time()
      # Training step
      print(f"Time per batch: {(time.time() - start):.3f}s")

***********************
RNA Design Issues
***********************

**Problem: RNA Design Not Converging**

**Symptoms:**

.. code-block:: text

   Generation 100: Best score = 5 (Hamming distance from target)
   No perfect matches found

**Solutions:**

1. **Increase Population Size**:

   .. code-block:: bash
   
      ogb rna_design --structure "(((...)))" --num-population 500

2. **Increase Generations**:

   .. code-block:: bash
   
      ogb rna_design --structure "(((...)))" --num-generation 300

3. **Adjust Mutation Rate**:

   .. code-block:: bash
   
      # Try lower mutation rate for fine-tuning
      ogb rna_design --structure "(((...)))" --mutation-ratio 0.2

4. **Use Different Model**:

   .. code-block:: bash
   
      # Try RNA-specialized model
      ogb rna_design --structure "(((...)))" --model yangheng/OmniGenome-186M

5. **Verify Structure Validity**:

   .. code-block:: python
   
      def validate_structure(structure):
          """Check if dot-bracket notation is balanced."""
          stack = []
          for char in structure:
              if char == '(':
                  stack.append(char)
              elif char == ')':
                  if not stack:
                      return False
                  stack.pop()
          return len(stack) == 0
      
      print(validate_structure("(((...)))"))  # True

**Problem: ViennaRNA Not Found**

**Symptoms:**

.. code-block:: text

   ModuleNotFoundError: No module named 'RNA'
   ImportError: cannot import name 'RNA'

**Solutions:**

1. **Install ViennaRNA**:

   .. code-block:: bash
   
      conda install -c bioconda viennarna

2. **Verify Installation**:

   .. code-block:: python
   
      import RNA
      print(RNA.fold("GCGAAACGC"))

***********************
Common Questions (FAQ)
***********************

**Q: How do I know which trainer backend to use?**

A: Follow these guidelines:

- **Development/Debugging**: Use ``trainer="native"`` for explicit control
- **Production Training**: Use ``trainer="accelerate"`` for multi-GPU scaling
- **Advanced Features**: Use ``trainer="hf_trainer"`` for DeepSpeed, callbacks

**Q: What's the difference between ``predict()`` and ``inference()``?**

A: 

- ``predict()``: Returns raw model outputs (logits, hidden states)
- ``inference()``: Returns formatted predictions with probabilities and class labels

**Q: How many seeds should I use for benchmarking?**

A: 

- **Quick experiments**: 1 seed
- **Paper results**: 3-5 seeds (recommended)
- **Critical comparisons**: 10+ seeds with significance testing

**Q: Why is my model loading slow?**

A: First-time loading downloads from HuggingFace Hub. Subsequent runs use cached models.

**Q: Can I use OmniGenBench for protein sequences?**

A: Currently optimized for DNA/RNA. Protein support is experimental.

**Q: How do I contribute to OmniGenBench?**

A: See ``CONTRIBUTING.md`` in the repository root.

***********************
Getting Help
***********************

If your issue persists after trying these solutions:

1. **Check GitHub Issues**: `<https://github.com/yangheng95/OmniGenBench/issues>`_
2. **Open New Issue**: Include error messages, system info, minimal reproduction code
3. **Documentation**: :doc:`index`
4. **API Reference**: :doc:`api_reference`

**System Information Template** (include when reporting issues):

.. code-block:: bash

   python -c "import omnigenbench; print(omnigenbench.__version__)"
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
   python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
   nvidia-smi  # If using GPU
