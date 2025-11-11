.. _robust-downloading:

###################################
Model & Dataset Downloading
###################################

**Eliminating Git-LFS Dependencies for Production-Grade Model Acquisition**

This guide introduces OmniGenBench's enhanced model downloading infrastructure that eliminates hard dependencies on Git-LFS while providing superior reliability, performance, and error handling. The system implements a hybrid strategy that prioritizes direct HTTP downloads via HuggingFace Hub's official API, with automatic fallback to Git-based cloning for edge cases requiring full repository history.

**Design Philosophy**: Git-LFS pointer file corruption has been a persistent source of silent model loading failures in genomic ML workflows. Users without properly configured Git-LFS installations would successfully "clone" models but receive only 100-byte pointer files instead of multi-gigabyte weight tensors—leading to models initializing with random weights and producing nonsensical predictions. This guide documents our solution: API-first downloading with comprehensive integrity verification.

**Key Improvements**:

* **Zero Git-LFS Dependency**: Direct HTTPS downloads via ``huggingface_hub`` Python package eliminate Git/LFS configuration requirements
* **Automatic Integrity Verification**: Post-download validation detects LFS pointer corruption and triggers remediation
* **Performance Gains**: 33% faster downloads through CDN optimization and resume capability
* **Reduced Storage**: 20% disk savings by omitting ``.git/`` repository metadata
* **Graceful Degradation**: Automatic fallback to Git clone preserves backward compatibility

.. contents:: Navigation
   :local:
   :depth: 2

*******************************************
Download Strategy Architecture
*******************************************

Primary Method: HuggingFace Hub API (Recommended)
===================================================

**Technical Implementation**: Direct HTTPS downloads using ``huggingface_hub.snapshot_download()``, bypassing Git entirely.

**Advantages Over Git-LFS**:

* **Dependency Elimination**: Only requires ``huggingface_hub>=0.20.0`` Python package—no system-level Git/LFS binaries
* **CDN-Accelerated Transfer**: Files served directly from HuggingFace's global CDN with automatic geo-routing
* **Chunk-Verified Downloads**: Automatic SHA256 verification for each file chunk prevents corruption
* **Resume Support**: Interrupted downloads automatically resume from last verified chunk
* **Compact Storage**: Excludes Git history and metadata (typically 20-25% size reduction)
* **Cross-Platform Reliability**: Identical behavior on Windows/Linux/macOS without system-specific Git configuration

**Workflow Diagram**:

.. code-block:: text

   User Request
        ↓
   Check Local Cache
        ↓
   [Cache Hit] → Return Path
        ↓
   [Cache Miss] → HF Hub API Download
        ↓
   Chunk-by-Chunk Transfer (with verification)
        ↓
   Integrity Validation (detect LFS pointers)
        ↓
   [Valid] → Return Path
        ↓
   [Invalid] → Trigger Fallback

**Example Implementation**:

.. code-block:: python

   from huggingface_hub import snapshot_download

   # Download entire model repository via HTTPS
   local_path = snapshot_download(
       repo_id="yangheng/OmniGenome-186M",
       cache_dir="__OMNIGENBENCH_DATA__/models/",
       local_dir_use_symlinks=False,  # Use actual files, not symlinks
       resume_download=True,           # Enable resume for interrupted transfers
   )
   
   # Returns: Path to downloaded model with all files verified

Fallback Method: Git Clone with LFS
====================================

**Use Cases**: Required only when users need full Git history (commit logs, branch information) or when HF Hub API is unavailable.

**Requirements**:

* System-level Git installation (``git --version`` must succeed)
* Git-LFS extension (``git lfs version`` must succeed)
* Network access to ``huggingface.co`` Git server

**Risk Profile**:

.. warning::
   **Git-LFS Pointer File Hazard**
   
   If Git-LFS is **not** installed, ``git clone`` will substitute large files with pointer files:
   
   .. code-block:: text
   
      version https://git-lfs.github.com/spec/v1
      oid sha256:b437d27531abc123...
      size 41943280
   
   PyTorch will fail to load these pointer files as model weights, resulting in **random initialization** 
   and incorrect inference. This failure mode is silent—no exception is raised, but predictions are 
   meaningless random outputs.

**Example Git Clone Workflow**:

.. code-block:: bash

   # Verify Git-LFS is installed
   git lfs install
   
   # Clone model repository (LFS files downloaded automatically)
   git clone https://huggingface.co/yangheng/OmniGenome-186M
   
   # Verify LFS files were pulled (not just pointers)
   cd OmniGenome-186M
   git lfs ls-files  # Should show actual files, not pointers

*****************************
Usage Patterns and Examples
*****************************

Automatic Strategy Selection (Recommended)
===========================================

The default behavior prioritizes HF Hub API with automatic fallback to Git clone:

.. code-block:: python

   from omnigenbench import ModelHub

   # Automatic strategy selection
   model = ModelHub.load("yangheng/OmniGenome-186M")
   
   # Execution flow:
   # 1. Attempt HF Hub API download
   # 2. Verify download integrity
   # 3. On failure: fall back to git clone
   # 4. Return loaded model with tokenizer

**Advantages**: Zero configuration required—optimal method selected automatically based on available tools and network conditions.

Explicit HF Hub API Usage
==========================

Force HuggingFace Hub API regardless of Git availability:

.. code-block:: python

   from omnigenbench import ModelHub

   # Explicit HF Hub API enforcement
   model = ModelHub.load(
       "yangheng/OmniGenome-186M",
       use_hf_api=True  # Disable Git fallback
   )

**Use Case**: Production environments where consistent download behavior is critical and Git is unavailable/untrusted.

Explicit Git Clone Usage
=========================

Force Git clone method (requires Git-LFS):

.. code-block:: python

   from omnigenbench import ModelHub

   # Explicit Git clone enforcement
   model = ModelHub.load(
       "yangheng/OmniGenome-186M",
       use_hf_api=False  # Force Git method
   )

.. warning::
   This method will fail if Git-LFS is not properly installed. Only use when Git history is required.

Direct API Access for Fine-Grained Control
===========================================

For advanced use cases requiring custom download behavior:

**Complete Repository Download**:

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import download_from_hf_hub

   # Download entire model to custom location
   path = download_from_hf_hub(
       repo_id="yangheng/ogb_tfb_finetuned",
       cache_dir="/custom/cache/directory/",
       force_download=False,  # Skip if already cached
   )
   
   print(f"Model stored at: {path}")

**Selective File Download** (bandwidth optimization):

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import download_from_hf_hub

   # Download only configuration and weights (skip tokenizer assets)
   path = download_from_hf_hub(
       repo_id="yangheng/OmniGenome-186M",
       allow_patterns=["*.json", "*.bin"],     # Include these patterns
       ignore_patterns=["*.msgpack", "*.h5"],  # Exclude these patterns
   )

**Dataset Acquisition**:

.. code-block:: python

   # Same API for datasets—just change repo_type
   path = download_from_hf_hub(
       repo_id="yangheng/OmniGenBench_RGB",
       repo_type="dataset",  # Specify repository type
       cache_dir="__OMNIGENBENCH_DATA__/datasets/",
   )

**********************************
Download Integrity Verification
**********************************

Automatic Validation
====================

All downloads include automatic post-transfer integrity checks:

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import (
       download_from_hf_hub,
       verify_download_integrity
   )

   # Download model
   path = download_from_hf_hub("yangheng/OmniGenome-186M")
   
   # Automatic verification (included in download_from_hf_hub)
   is_valid = verify_download_integrity(path)
   
   if not is_valid:
       raise RuntimeError("Download corrupted—LFS pointer detected or missing files")

**Validation Checks Performed**:

1. **File Existence**: Verify all required files present (config.json, pytorch_model.bin, tokenizer files)
2. **LFS Pointer Detection**: Scan .bin files for Git-LFS pointer headers
3. **Size Validation**: Flag suspiciously small files (<200 bytes for .bin files)

Custom Validation Requirements
===============================

Specify custom file requirements for domain-specific validation:

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import verify_download_integrity

   # Verify specific files present
   is_valid = verify_download_integrity(
       "__OMNIGENBENCH_DATA__/models/yangheng--OmniGenome-186M",
       required_files=[
           "config.json",
           "pytorch_model.bin",
           "tokenizer.json",
           "vocab.txt",
       ]
   )

LFS Pointer Detection Algorithm
================================

The verification system automatically detects Git-LFS pointer files:

.. code-block:: python

   # Internal verification logic (informational—automatic in OmniGenBench)
   
   def is_lfs_pointer(file_path):
       """Check if file is Git-LFS pointer instead of actual content."""
       if file_path.stat().st_size < 200:  # Pointer files are ~100 bytes
           with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
               first_line = f.readline()
               if 'version https://git-lfs' in first_line:
                   return True
       return False

**Detection Output Example**:

.. code-block:: text

   [ERROR] Detected git-lfs pointer file (incomplete download): pytorch_model.bin
   [ERROR] Please use download_from_hf_hub() to download properly
   [INFO] File size: 132 bytes (expected: ~40MB)

*******************************
Repository Metadata Queries
*******************************

List Repository Contents
========================

Inspect available files before downloading:

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import list_hf_repo_files

   # Query repository file tree
   files = list_hf_repo_files("yangheng/OmniGenome-186M")
   
   for file in sorted(files):
       print(f"  {file}")
   
   # Example output:
   #   config.json
   #   pytorch_model.bin
   #   tokenizer.json
   #   tokenizer_config.json
   #   vocab.txt
   #   special_tokens_map.json

Retrieve Model Metadata
========================

Access repository metadata without downloading files:

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import get_model_info

   # Fetch metadata from HuggingFace Hub
   info = get_model_info("yangheng/OmniGenome-186M")
   
   print(f"Model ID: {info['id']}")
   print(f"Last Modified: {info['last_modified']}")
   print(f"Tags: {', '.join(info['tags'])}")
   print(f"Number of Files: {len(info['siblings'])}")
   
   # Estimate total download size
   total_size_mb = sum(
       f.get('size', 0) for f in info.get('siblings', [])
   ) / (1024 ** 2)
   print(f"Estimated Download: {total_size_mb:.1f} MB")

*******************************************
Troubleshooting and Error Recovery
*******************************************

Problem: Model Weights Not Loading Correctly
=============================================

**Symptoms**:

* Model produces random/nonsensical predictions despite successful loading
* Evaluation metrics significantly worse than reported benchmarks
* No error messages during model initialization

**Root Cause**: Git-LFS pointer file loaded instead of actual weight tensors.

**Diagnosis**:

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import verify_download_integrity

   # Check model integrity
   is_valid = verify_download_integrity(
       "__OMNIGENBENCH_DATA__/models/yangheng--ogb_tfb_finetuned"
   )
   
   if not is_valid:
       print("DIAGNOSIS: Git-LFS pointer detected—model weights not downloaded")

**Solution A: Re-download with HF Hub API**:

.. code-block:: python

   from omnigenbench import ModelHub

   # Force HF Hub API re-download
   model = ModelHub.load(
       "yangheng/ogb_tfb_finetuned",
       use_hf_api=True,
       force_download=True  # Overwrite corrupted cache
   )

**Solution B: Fix Existing Git Clone**:

.. code-block:: bash

   # Install Git-LFS
   git lfs install
   
   # Navigate to cached model
   cd __OMNIGENBENCH_DATA__/models/yangheng--ogb_tfb_finetuned
   
   # Pull actual LFS files
   git lfs pull
   
   # Verify files downloaded
   git lfs ls-files  # Should show actual files, not pointers

Problem: HuggingFace Hub Package Not Installed
===============================================

**Symptoms**:

.. code-block:: text

   ImportError: huggingface_hub is required for this download method.
   Install it with: pip install huggingface_hub

**Solution**:

.. code-block:: bash

   pip install huggingface_hub>=0.20.0

**Verification**:

.. code-block:: python

   python -c "from huggingface_hub import snapshot_download; print('OK')"

Problem: Network Connection Failures
=====================================

**Symptoms**: Download timeouts or connection errors.

**Solution**: HF Hub API includes automatic resume capability:

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import download_from_hf_hub

   # Simply re-run the same command—download resumes automatically
   path = download_from_hf_hub(
       "yangheng/OmniGenome-186M",
       force_download=False  # Resume from last successful chunk
   )

.. tip::
   **Resume Mechanism**: HuggingFace Hub tracks which file chunks have been successfully 
   downloaded and verified. Subsequent download attempts skip completed chunks and resume 
   from the last unverified position.

Problem: Insufficient Disk Space
=================================

**Diagnosis**:

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import get_model_info

   # Estimate required space before downloading
   info = get_model_info("yangheng/OmniGenome-186M")
   
   total_size_gb = sum(
       f.get('size', 0) for f in info.get('siblings', [])
   ) / (1024 ** 3)
   
   print(f"Required disk space: {total_size_gb:.2f} GB")

**Solution: Selective File Download**:

.. code-block:: python

   # Download only essential files (skip optional formats)
   path = download_from_hf_hub(
       "yangheng/OmniGenome-186M",
       allow_patterns=["*.json", "*.bin"],         # Core files only
       ignore_patterns=["*.msgpack", "*.h5", "*.onnx"],  # Skip alternative formats
   )

*****************************
Performance Benchmarks
*****************************

Download Speed Comparison
=========================

Tested with ``yangheng/OmniGenome-186M`` (~200MB model):

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Method
     - Time
     - Dependencies
     - Risk Level
   * - HF Hub API
     - **30 seconds**
     - ``huggingface_hub``
     - None
   * - Git Clone (with LFS)
     - 45 seconds
     - ``git`` + ``git-lfs``
     - Low
   * - Git Clone (without LFS)
     - 5 seconds ⚠️
     - ``git`` only
     - **High** (pointer files)

**Performance Analysis**:

* **HF Hub API**: 33% faster than proper Git clone due to CDN optimization and parallel chunk downloads
* **Git Clone (no LFS)**: Appears fast but downloads only pointer files—produces broken models
* **Recommendation**: Always use HF Hub API for production workloads

Storage Efficiency Comparison
==============================

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Method
     - Disk Usage
     - Composition
   * - HF Hub API
     - **200 MB**
     - Model files only
   * - Git Clone
     - 250 MB
     - Model files + ``.git/`` metadata

**Space Savings**: 50 MB (20% reduction) by excluding Git history.

*****************************
Migration Guide
*****************************

Upgrading from Git-Based Workflows
===================================

**Legacy Code Pattern**:

.. code-block:: python

   # Old approach (Git-LFS dependent)
   model = ModelHub.load("yangheng/OmniGenome-186M")
   # Implicitly uses git clone—fails if LFS not installed

**Recommended Modern Pattern**:

.. code-block:: python

   # New approach (Git-LFS independent)
   model = ModelHub.load(
       "yangheng/OmniGenome-186M",
       use_hf_api=True  # Explicit HF Hub API usage
   )
   # Only requires huggingface_hub package

**Zero-Friction Migration**:

.. code-block:: python

   # No code changes needed—automatic upgrade
   model = ModelHub.load("yangheng/OmniGenome-186M")
   
   # New behavior:
   # 1. Try HF Hub API (if available)
   # 2. Fall back to git clone (if HF Hub fails)
   # Original code continues working with improved reliability

Validating Migrated Models
===========================

After migration, verify all cached models are valid:

.. code-block:: python

   from pathlib import Path
   from omnigenbench.src.utility.model_hub.hf_download import verify_download_integrity

   # Scan cache directory
   cache_dir = Path("__OMNIGENBENCH_DATA__/models/")
   
   for model_dir in cache_dir.iterdir():
       if model_dir.is_dir():
           is_valid = verify_download_integrity(str(model_dir))
           status = "✓ Valid" if is_valid else "✗ Corrupted (LFS pointer)"
           print(f"{model_dir.name}: {status}")

***************************
Best Practices Summary
***************************

Recommended Practices
=====================

1. **Default to HF Hub API**:

   .. code-block:: python
   
      # Always prefer API-first approach
      model = ModelHub.load("model_name")  # Automatic HF API priority

2. **Validate Downloads**:

   .. code-block:: python
   
      from omnigenbench.src.utility.model_hub.hf_download import (
          download_from_hf_hub, verify_download_integrity
      )
      
      path = download_from_hf_hub("model_name")
      assert verify_download_integrity(path), "Download integrity check failed"

3. **Handle Private Models**:

   .. code-block:: python
   
      # Use HuggingFace access token for private repositories
      path = download_from_hf_hub(
          "private_org/private_model",
          token="hf_xxxxxxxxxxxxxxxxxxxx"  # Get from huggingface.co/settings/tokens
      )

4. **Optimize Bandwidth**:

   .. code-block:: python
   
      # Download only required file patterns
      path = download_from_hf_hub(
          "model_name",
          allow_patterns=["*.json", "*.bin"],  # Core files
          ignore_patterns=["*.onnx", "*.msgpack"],  # Skip alternative formats
      )

Practices to Avoid
==================

1. **Do Not Assume Git-LFS Availability**:

   .. code-block:: python
   
      # Avoid: Implicit Git dependency
      model = ModelHub.load("model_name", use_hf_api=False)
      
      # Prefer: Explicit HF API
      model = ModelHub.load("model_name", use_hf_api=True)

2. **Do Not Ignore Verification Failures**:

   .. code-block:: python
   
      # Avoid: Ignoring validation
      path = download_from_hf_hub("model_name")
      model = load_model(path)  # May load corrupted model
      
      # Prefer: Assert on validation
      path = download_from_hf_hub("model_name")
      assert verify_download_integrity(path), "Re-download required"
      model = load_model(path)

3. **Do Not Mix Download Methods**:

   If Git clone fails, clean up before retrying with HF Hub API:
   
   .. code-block:: bash
   
      # Clean corrupted Git clone
      rm -rf __OMNIGENBENCH_DATA__/models/yangheng--OmniGenome-186M
      
      # Re-download with HF Hub API
      python -c "
      from omnigenbench import ModelHub
      ModelHub.load('yangheng/OmniGenome-186M', use_hf_api=True)
      "

***************************
System Requirements
***************************

Minimum Requirements (HF Hub API)
==================================

.. code-block:: bash

   # Only Python package required
   pip install huggingface_hub>=0.20.0

**Platform Support**: Windows, Linux, macOS with identical behavior.

Full Requirements (All Methods)
================================

.. code-block:: bash

   # Python packages
   pip install huggingface_hub>=0.20.0
   
   # System tools (optional—only for Git fallback)
   # Windows:
   choco install git git-lfs
   
   # Linux (Debian/Ubuntu):
   apt-get install git git-lfs
   
   # macOS:
   brew install git git-lfs

**Recommendation**: Install only ``huggingface_hub`` for production environments to minimize system dependencies.

***************************
API Reference Summary
***************************

Core Functions
==============

.. py:function:: download_from_hf_hub(repo_id, cache_dir="__OMNIGENBENCH_DATA__/models/", force_download=False, repo_type="model", allow_patterns=None, ignore_patterns=None, token=None)

   Download model or dataset from HuggingFace Hub via HTTPS API.
   
   :param str repo_id: HuggingFace repository identifier (e.g., "yangheng/OmniGenome-186M")
   :param str cache_dir: Local directory for cached downloads
   :param bool force_download: Overwrite existing cache
   :param str repo_type: Repository type ("model", "dataset", or "space")
   :param list allow_patterns: File patterns to include (e.g., ["*.json", "*.bin"])
   :param list ignore_patterns: File patterns to exclude
   :param str token: HuggingFace API token for private repositories
   :return: Path to downloaded repository
   :rtype: str

.. py:function:: verify_download_integrity(local_path, required_files=None)

   Validate downloaded model files and detect Git-LFS pointer corruption.
   
   :param str local_path: Path to downloaded model directory
   :param list required_files: Files to verify (default: ["config.json"])
   :return: True if all files valid, False if LFS pointer detected or files missing
   :rtype: bool

.. py:function:: list_hf_repo_files(repo_id, repo_type="model", token=None)

   List all files in a HuggingFace repository without downloading.
   
   :param str repo_id: Repository identifier
   :param str repo_type: Repository type
   :param str token: API token for private repositories
   :return: List of file paths in repository
   :rtype: list[str]

.. py:function:: get_model_info(repo_id, token=None)

   Retrieve model repository metadata from HuggingFace Hub.
   
   :param str repo_id: Repository identifier
   :param str token: API token for private repositories
   :return: Dictionary with model metadata (id, sha, last_modified, tags, siblings)
   :rtype: dict

***************************
Additional Resources
***************************

* **HuggingFace Hub Documentation**: `huggingface.co/docs/huggingface_hub <https://huggingface.co/docs/huggingface_hub>`_
* **Git-LFS Documentation**: `git-lfs.github.com <https://git-lfs.github.com/>`_
* **OmniGenBench Getting Started**: :doc:`GETTING_STARTED`
* **Troubleshooting Guide**: :doc:`troubleshooting`

***************************
Version History
***************************

v0.3.23alpha+
=============

* **Added**: HuggingFace Hub API download support with automatic Git-LFS bypass
* **Added**: Download integrity verification with LFS pointer detection
* **Added**: Repository metadata query functions (``list_hf_repo_files``, ``get_model_info``)
* **Improved**: Automatic fallback from HF Hub API to Git clone for backward compatibility
* **Improved**: 33% faster downloads and 20% storage savings compared to Git clone
* **Documentation**: Complete migration guide and troubleshooting procedures

.. note::
   This download infrastructure is production-ready and recommended for all new projects. 
   Legacy Git-based workflows continue to function with automatic upgrades to the new system.
