Trainers
========

.. note::
   
   **You are viewing the API reference documentation.**
   
   This page provides detailed API documentation for trainer classes. For a comprehensive tutorial-style guide with complete examples, see the :doc:`../TRAINER_GUIDE`.
   
   - **Quick Start**: See below for minimal examples
   - **Detailed Tutorial**: :doc:`../TRAINER_GUIDE` (complete guide with 10 sections)
   - **Design Philosophy**: :doc:`../design_principle` (understanding BaseTrainer abstraction)

Overview
--------

OmniGenBench provides three powerful trainer implementations for different training scenarios. All trainers inherit from a unified ``BaseTrainer`` abstract class, ensuring consistent API and functionality.

Trainer Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Trainer
     - Best For
     - Key Features
     - Requirements
   * - **Trainer**
     - Single-GPU training
     - Lightweight, easy debugging, native PyTorch
     - PyTorch 2.5+
   * - **AccelerateTrainer**
     - Multi-GPU distributed training
     - Zero-config distributed, DeepSpeed/FSDP support
     - HuggingFace Accelerate
   * - **HFTrainer**
     - HuggingFace ecosystem integration
     - Full HF features, callbacks, logging
     - HuggingFace Transformers

Quick Start
-----------

**Native Trainer** (Single-GPU)::

    from omnigenbench import Trainer, ModelHub
    
    model = ModelHub.load("yangheng/OmniGenome-186M")
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=10,
        batch_size=32,
        autocast="fp16",  # Mixed precision
        device="cuda:0",
    )
    
    metrics = trainer.train()

**Accelerate Trainer** (Multi-GPU)::

    from omnigenbench import AccelerateTrainer
    
    trainer = AccelerateTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=10,
        batch_size=32,  # Per-device batch size
        autocast="fp16",
    )
    
    metrics = trainer.train()
    
    # Run with: accelerate launch train.py

**HuggingFace Trainer**::

    from omnigenbench import HFTrainer
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        fp16=True,
    )
    
    trainer = HFTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    metrics = trainer.train()

Common Features
---------------

All trainers support:

- ✅ **Mixed Precision Training**: FP16, BF16 for faster training
- ✅ **Early Stopping**: Automatic training termination when metrics plateau
- ✅ **Gradient Accumulation**: Simulate larger batch sizes
- ✅ **Gradient Clipping**: Prevent exploding gradients
- ✅ **Checkpoint Management**: Automatic saving and loading
- ✅ **Flexible Optimizers**: Support any PyTorch optimizer
- ✅ **Learning Rate Scheduling**: Built-in and custom schedulers
- ✅ **Evaluation Metrics**: Custom metric computation
- ✅ **Reproducibility**: Seed control for consistent results

Base Trainer
------------

Abstract base class defining the common interface for all trainers.

.. automodule:: omnigenbench.src.trainer.base_trainer
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Key Components
~~~~~~~~~~~~~~

**MetricsDict**

Enhanced dictionary for training metrics with formatted display::

    metrics = trainer.train()
    print(metrics)  # Automatically formatted output
    
    # Access metrics
    best_accuracy = metrics['best_valid']['eval_accuracy']
    final_loss = metrics['train_metrics_history'][-1]['train_loss']

**Optimization Direction**

Automatically infers whether to minimize or maximize metrics::

    trainer = Trainer(
        optimization_metric="loss",      # Minimizes
        # or
        optimization_metric="accuracy",  # Maximizes
    )

Trainer (Native PyTorch)
------------------------

Native PyTorch trainer for single-GPU training.

.. automodule:: omnigenbench.src.trainer.trainer
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Features
~~~~~~~~

- **Automatic Mixed Precision**: Using ``torch.cuda.amp.GradScaler``
- **Device Management**: Auto-detects CUDA/CPU
- **Simple and Fast**: Minimal overhead, easy debugging
- **Customizable**: Easy to extend and modify

Example
~~~~~~~

::

    from omnigenbench import Trainer
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # Configure optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=10000
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        
        # Training configuration
        epochs=10,
        batch_size=32,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        
        # Mixed precision
        autocast="fp16",  # or "bf16" for Ampere+ GPUs
        
        # Early stopping
        patience=3,
        delta=0.001,
        
        # Checkpointing
        save_dir="./checkpoints",
        save_steps=1000,
        save_total_limit=3,
        
        # Device
        device="cuda:0",
        seed=42,
    )
    
    # Train
    metrics = trainer.train()
    
    # Evaluate
    test_metrics = trainer.evaluate(test_dataset)

AccelerateTrainer (Distributed Training)
-----------------------------------------

Distributed trainer using HuggingFace Accelerate for multi-GPU training.

.. automodule:: omnigenbench.src.trainer.accelerate_trainer
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Features
~~~~~~~~

- **Zero-Config Distributed**: Automatic multi-GPU detection and setup
- **DeepSpeed Integration**: Support for Zero Stage 1/2/3
- **FSDP Support**: Fully Sharded Data Parallel
- **Flexible Backends**: DDP, DeepSpeed, FSDP, and more
- **Gradient Synchronization**: Automatic gradient accumulation across devices

Distributed Training Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-GPU (DDP)**::

    # Auto-detects all GPUs
    trainer = AccelerateTrainer(
        model=model,
        train_dataset=train_dataset,
        batch_size=32,  # Per-device batch size
    )
    
    # Launch
    accelerate launch train.py

**DeepSpeed**::

    # Create ds_config.json
    {
      "train_batch_size": "auto",
      "gradient_accumulation_steps": "auto",
      "zero_optimization": {
        "stage": 2
      },
      "fp16": {
        "enabled": true
      }
    }
    
    # Launch
    accelerate launch --config_file ds_config.json train.py

**FSDP (Large Models)**::

    # Configure once
    accelerate config  # Select FSDP, FULL_SHARD
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    trainer = AccelerateTrainer(
        model=model,
        train_dataset=train_dataset,
        batch_size=16,
    )
    
    # Launch
    accelerate launch train.py

Example
~~~~~~~

::

    from omnigenbench import AccelerateTrainer
    
    trainer = AccelerateTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        
        epochs=10,
        batch_size=16,  # Per-device batch size
        gradient_accumulation_steps=4,
        
        autocast="fp16",
        
        save_dir="./checkpoints",
        eval_steps=500,
        save_steps=500,
    )
    
    # Train
    metrics = trainer.train()
    
    # Only print on main process
    if trainer.accelerator.is_main_process:
        print(metrics)

HFTrainer (HuggingFace Integration)
------------------------------------

Wrapper for HuggingFace Trainer with OmniGenome metadata support.

.. automodule:: omnigenbench.src.trainer.hf_trainer
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Features
~~~~~~~~

- **Full HF Ecosystem**: Seamless integration with Transformers
- **Rich Callbacks**: Built-in callbacks for WandB, TensorBoard, etc.
- **Advanced Logging**: Comprehensive training logs
- **Checkpoint Management**: Automatic best model tracking
- **Custom Metrics**: Easy metric computation

Example
~~~~~~~

::

    from omnigenbench import HFTrainer
    from transformers import TrainingArguments
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    
    # Define metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average='weighted'),
        }
    
    # Configure training
    training_args = TrainingArguments(
        output_dir="./results",
        
        # Training
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=500,
        
        # Saving
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        
        # Logging
        logging_dir="./logs",
        logging_steps=100,
        report_to=["tensorboard", "wandb"],
        
        # Mixed precision
        fp16=True,
        
        # Other
        seed=42,
    )
    
    # Create trainer
    trainer = HFTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    train_result = trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    test_results = trainer.evaluate(test_dataset)
    
    # Save
    trainer.save_model("./final_model")
    trainer.save_metrics("test", test_results)

Advanced Usage
--------------

Custom Trainer
~~~~~~~~~~~~~~

Extend ``BaseTrainer`` for custom training logic::

    from omnigenbench.src.trainer import BaseTrainer
    import torch
    
    class CustomTrainer(BaseTrainer):
        def _setup_training_components(self):
            """Setup custom components"""
            self.device = torch.device("cuda")
            self.model.to(self.device)
            self.scaler = torch.cuda.amp.GradScaler()
        
        def _prepare_batch(self, batch):
            """Custom batch preparation"""
            return batch.to(self.device)
        
        def _train_epoch(self, epoch):
            """Custom training loop"""
            self.model.train()
            total_loss = 0
            
            for batch in self.train_loader:
                batch = self._prepare_batch(batch)
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
            
            return total_loss / len(self.train_loader)

Multi-Task Learning
~~~~~~~~~~~~~~~~~~~~

Train on multiple tasks simultaneously::

    class MultiTaskTrainer(Trainer):
        def __init__(self, model, task_datasets, task_weights=None, **kwargs):
            self.task_datasets = task_datasets
            self.task_weights = task_weights or {t: 1.0 for t in task_datasets}
            super().__init__(model, **kwargs)
        
        def _train_epoch(self, epoch):
            self.model.train()
            total_loss = 0
            
            for task_name, dataset in self.task_datasets.items():
                task_weight = self.task_weights[task_name]
                # ... train on each task
            
            return total_loss

Gradient Checkpointing
~~~~~~~~~~~~~~~~~~~~~~

Save memory for large models::

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    trainer = AccelerateTrainer(
        model=model,
        batch_size=64,  # Can use larger batches
    )

Best Practices
--------------

✅ **Recommended**

- Set random seed for reproducibility
- Use mixed precision (FP16/BF16) for faster training
- Enable early stopping to prevent overfitting
- Save checkpoints regularly
- Monitor both training and validation metrics
- Use gradient accumulation for large effective batch sizes

❌ **Avoid**

- Training without validation set
- Extremely large learning rates
- Not saving checkpoints (risk losing progress)
- Ignoring GPU memory constraints
- Training without monitoring metrics

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

::

    # 1. Efficient data loading
    trainer = Trainer(
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    
    # 2. PyTorch 2.0 compilation
    model = torch.compile(model)
    
    # 3. Gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # 4. Mixed precision
    trainer = Trainer(autocast="bf16")  # More stable than fp16
    
    # 5. DeepSpeed for large-scale training
    # See AccelerateTrainer documentation

See Also
--------

- :doc:`../TRAINER_GUIDE` - Comprehensive guide with examples
- :doc:`../usage` - Basic usage examples
- :doc:`datasets` - Dataset documentation
- :doc:`models` - Model documentation

External Resources
~~~~~~~~~~~~~~~~~~

- `HuggingFace Trainer <https://huggingface.co/docs/transformers/main_classes/trainer>`_
- `HuggingFace Accelerate <https://huggingface.co/docs/accelerate/>`_
- `DeepSpeed <https://www.deepspeed.ai/>`_
- `PyTorch FSDP <https://pytorch.org/docs/stable/fsdp.html>`_