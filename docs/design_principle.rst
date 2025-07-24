Design Principles
=====================


OmniGenBench is designed to provide a unified, extensible, and robust framework for genomic foundation models. The core philosophy centers on abstraction, modularity, and interoperability, enabling users to build, extend, and integrate genomic models and data pipelines with minimal friction.

Definitions of Abstract Classes
-------------------------------
Abstract base classes are fundamental to OmniGenBench's architecture. They define clear contracts for models, datasets, tokenizers, and metrics, ensuring that all components follow consistent interfaces. This approach offers several advantages:

- **Consistency**: All implementations adhere to the same interface, reducing bugs and confusion.
- **Extensibility**: Users can easily extend functionality by subclassing abstract classes.
- **Interoperability**: Components can be swapped or combined without breaking the workflow.
- **Maintainability**: Code is easier to maintain and update as new features are added.

How to Extend Abstract Classes
------------------------------
To add new functionality, simply subclass the relevant abstract class and implement the required methods. This allows you to create custom models, datasets, tokenizers, or metrics tailored to your specific needs.

**Example: Custom Model Extension**

.. code-block:: python

    from omnigenbench import OmniModel
    import torch

    class CustomGenomicModel(OmniModel):
        def __init__(self, config_or_model, tokenizer, **kwargs):
            super().__init__(config_or_model, tokenizer, **kwargs)
            self.custom_layer = torch.nn.Linear(self.config.hidden_size, self.num_labels)

        def forward(self, **inputs):
            outputs = self.last_hidden_state_forward(**inputs)
            logits = self.custom_layer(outputs.last_hidden_state)
            if 'labels' in inputs:
                loss = self.compute_loss(logits, inputs['labels'])
                return type(outputs)(loss=loss, logits=logits)
            return type(outputs)(logits=logits)

**Example: Custom Dataset Extension**

.. code-block:: python

    from omnigenbench import OmniDatasetForSequenceClassification

    class CustomGenomicDataset(OmniDatasetForSequenceClassification):
        def __init__(self, data_path, tokenizer, **kwargs):
            super().__init__(data_path, tokenizer, **kwargs)

        def _load_data(self, data_path):
            data = self._load_json(data_path)
            return self._process_data(data)

        def _process_data(self, data):
            processed_data = []
            for item in data:
                processed_item = self._process_item(item)
                processed_data.append(processed_item)
            return processed_data

**Example: Custom Tokenizer Extension**

.. code-block:: python

    from omnigenbench import OmniTokenizer

    class CustomGenomicTokenizer(OmniTokenizer):
        def __init__(self, base_tokenizer, **kwargs):
            super().__init__(base_tokenizer, **kwargs)

        def tokenize(self, sequence, **kwargs):
            tokens = self._custom_tokenize(sequence)
            return [tokens]

        def _custom_tokenize(self, sequence):
            k = self.k if hasattr(self, 'k') else 3
            return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

**Example: Custom Metric Extension**
.. code-block:: python

    from omnigenbench import OmniMetric
    from sklearn.metrics import custom_metric

    class CustomGenomicMetric(OmniMetric):
        def __init__(self, ignore_y=None, **kwargs):
            super().__init__(ignore_y=ignore_y, **kwargs)
            self.metric_name = "custom_metric"

        def compute_metric(self, y_true, y_pred, **kwargs):
            mask = y_true != self.ignore_y
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            score = custom_metric(y_true_filtered, y_pred_filtered)
            return {self.metric_name: score}

Core Concepts and Patterns
--------------------------
- **Model-Data Integration**: Abstract classes are designed to work together seamlessly, enabling easy integration of models, datasets, tokenizers, and metrics.
- **Configuration Management**: All components support flexible configuration via keyword arguments and config dictionaries.
- **Error Handling**: Robust error handling is built into the abstract classes, providing meaningful messages for invalid inputs.
- **Performance**: The framework supports memory-efficient data handling, caching, parallelization, and GPU utilization.
- **Extension Points**: Users can override loss functions, preprocessing, metrics, tokenization, and data formats for custom workflows.

Best Practices
--------------
1. Always inherit from the appropriate abstract base class.
2. Implement all required abstract methods.
3. Provide comprehensive docstrings and examples.
4. Write unit tests for custom implementations.
5. Follow established patterns and conventions for consistency.

Summary
-------
OmniGenBench's design principles ensure that the framework is easy to use, extend, and maintain. By leveraging abstract classes and modular design, users can build powerful genomic analysis pipelines that are both robust and flexible.

Overview
--------

OmniGenBench is built around a set of core abstract base classes that provide a unified interface for working with genomic data and models. These abstract classes define the contract that all implementations must follow, ensuring consistency and interoperability across the framework.

Core Abstract Classes
---------------------

Abstract Model
~~~~~~~~~~~~~~

The ``OmniModel`` abstract base class serves as the foundation for all models in OmniGenBench. It provides a unified interface for model initialization, forward passes, and inference operations.

**Key Features:**

- **Unified Interface**: All models follow the same interface regardless of their underlying architecture
- **Flexible Initialization**: Supports initialization from pre-trained models, PyTorch modules, or configuration objects
- **Automatic Loss Computation**: Handles loss calculation for different task types automatically
- **Model Persistence**: Built-in support for saving and loading models
- **Inference Pipeline**: Standardized inference methods for easy deployment

**Core Methods:**

- ``__init__(config_or_model, tokenizer, **kwargs)``: Initialize the model
- ``forward(**inputs)``: Perform forward pass with automatic loss computation
- ``predict(sequence)``: Generate predictions for input sequences
- ``inference(sequence)``: Full inference pipeline with preprocessing and postprocessing
- ``save_model(path)``: Save model to disk
- ``load_model(path)``: Load model from disk

**Usage Example:**

.. code-block:: python

    from omnigenbench import OmniModelForSequenceClassification
    
    # Initialize model
    model = OmniModelForSequenceClassification("model_path", tokenizer)
    
    # Forward pass with labels (training)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    # Inference (evaluation)
    predictions = model.predict("ATCGATCG")
    print(predictions['predictions'])  # Class predictions
    print(predictions['confidence'])   # Confidence scores

Abstract Dataset
~~~~~~~~~~~~~~~~

The ``OmniDataset`` abstract base class provides a standardized interface for genomic datasets. It handles data loading, preprocessing, and provides a PyTorch-compatible dataset interface.

**Key Features:**

- **Multiple Format Support**: Handles CSV, JSON, Parquet, TXT, and other formats
- **Automatic Tokenization**: Integrates with tokenizers for seamless preprocessing
- **Label Mapping**: Automatic conversion between string labels and numeric indices
- **Data Validation**: Built-in validation for data integrity and format consistency
- **Flexible Configuration**: Configurable sequence length, padding, and truncation

**Core Methods:**

- ``__init__(data_path, tokenizer, **kwargs)``: Initialize dataset
- ``__getitem__(index)``: Get a single sample
- ``__len__()``: Return dataset size
- ``get_labels()``: Get unique labels in the dataset
- ``get_label_mapping()``: Get mapping between labels and indices

**Supported Data Formats:**

- **JSON**: ``{"sequence": "ATCG", "label": "positive"}``
- **CSV**: ``sequence,label\nATCG,positive``
- **Parquet**: Columnar format for large datasets
- **TXT**: Simple text files with one sequence per line

**Usage Example:**

.. code-block:: python

    from omnigenbench import OmniDatasetForSequenceClassification
    
    # Initialize dataset
    dataset = OmniDatasetForSequenceClassification(
        "data.json", 
        tokenizer, 
        max_length=512,
        label_column="label"
    )
    
    # Access data
    sample = dataset[0]
    print(sample['input_ids'].shape)      # torch.Size([512])
    print(sample['attention_mask'].shape) # torch.Size([512])
    print(sample['labels'])               # Label index
    
    # Get dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Labels: {dataset.get_labels()}")

Abstract Tokenizer
~~~~~~~~~~~~~~~~~~

The ``OmniTokenizer`` abstract base class provides a unified interface for tokenizing genomic sequences. It wraps different tokenization strategies and provides consistent preprocessing options.

**Key Features:**

- **Consistent Interface**: Same interface across different tokenization strategies
- **Custom Wrapper Support**: Easy integration with custom tokenizer implementations
- **Special Token Handling**: Automatic handling of BOS, EOS, and other special tokens
- **Sequence Preprocessing**: Options for U/T conversion, whitespace addition, and more
- **Flexible Configuration**: Configurable tokenization parameters

**Core Methods:**

- ``__init__(base_tokenizer, **kwargs)``: Initialize tokenizer
- ``tokenize(sequence, **kwargs)``: Tokenize input sequence
- ``encode(sequence, **kwargs)``: Encode sequence to token IDs
- ``decode(token_ids, **kwargs)``: Decode token IDs back to sequence
- ``from_pretrained(model_name)``: Load pre-trained tokenizer

**Preprocessing Options:**

- **U/T Conversion**: Convert U to T or vice versa
- **Whitespace Addition**: Add spaces between nucleotides
- **Case Normalization**: Convert to uppercase or lowercase
- **Special Token Handling**: Add BOS, EOS, PAD tokens automatically

**Usage Example:**

.. code-block:: python

    from omnigenbench import OmniSingleNucleotideTokenizer
    
    # Initialize tokenizer
    tokenizer = OmniSingleNucleotideTokenizer.from_pretrained("model_name")
    
    # Tokenize sequence
    inputs = tokenizer("ATCGATCG", max_length=512, padding=True)
    print(inputs['input_ids'].shape)      # torch.Size([1, 512])
    print(inputs['attention_mask'].shape) # torch.Size([1, 512])
    
    # Decode tokens
    decoded = tokenizer.decode(inputs['input_ids'][0])
    print(decoded)  # "ATCGATCG"

Abstract Metric
~~~~~~~~~~~~~~~

The ``OmniMetric`` abstract base class provides a standardized interface for evaluation metrics. It integrates with scikit-learn metrics and provides consistent result formatting.

**Key Features:**

- **Scikit-learn Integration**: Leverages scikit-learn's comprehensive metric collection
- **Ignored Label Support**: Handles special labels like -100 for ignored tokens
- **Flexible Input Formats**: Accepts various input formats (lists, arrays, tensors)
- **Consistent Results**: Standardized result format across all metrics
- **Multi-task Support**: Support for multiple evaluation tasks

**Core Methods:**

- ``__init__(ignore_y=None, **kwargs)``: Initialize metric
- ``compute_metric(y_true, y_pred, **kwargs)``: Compute metric values
- ``format_results(results)``: Format results consistently
- ``get_metric_name()``: Get metric name for identification

**Supported Metric Types:**

- **Classification**: Accuracy, F1-score, Precision, Recall, AUC
- **Regression**: MSE, MAE, R², RMSE, MAPE
- **Ranking**: NDCG, MAP, MRR, Precision@k

**Usage Example:**

.. code-block:: python

    from omnigenbench import ClassificationMetric
    
    # Initialize metric
    metric = ClassificationMetric(ignore_y=-100)
    
    # Compute metrics
    y_true = [0, 1, 2, -100, 1]  # -100 is ignored
    y_pred = [0, 1, 1, 0, 1]
    
    results = metric.compute_metric(y_true, y_pred)
    print(results)
    # {
    #     'accuracy_score': 0.75,
    #     'f1_score': 0.8,
    #     'precision_score': 0.75,
    #     'recall_score': 0.67
    # }

Implementation Patterns
-----------------------

Model Implementation
~~~~~~~~~~~~~~~~~~~~

When implementing a new model, inherit from the appropriate abstract base class:

.. code-block:: python

    from omnigenbench import OmniModel
    
    class CustomGenomicModel(OmniModel):
        def __init__(self, config_or_model, tokenizer, **kwargs):
            super().__init__(config_or_model, tokenizer, **kwargs)
            # Add custom layers
            self.custom_classifier = torch.nn.Linear(
                self.config.hidden_size, 
                self.num_labels
            )
        
        def forward(self, **inputs):
            # Get base model outputs
            outputs = self.last_hidden_state_forward(**inputs)
            
            # Apply custom classifier
            logits = self.custom_classifier(outputs.last_hidden_state)
            
            # Handle loss computation
            if 'labels' in inputs:
                loss = self.compute_loss(logits, inputs['labels'])
                return type(outputs)(loss=loss, logits=logits)
            
            return type(outputs)(logits=logits)

Dataset Implementation
~~~~~~~~~~~~~~~~~~~~~~

For custom datasets, inherit from the appropriate dataset base class:

.. code-block:: python

    from omnigenbench import OmniDatasetForSequenceClassification
    
    class CustomGenomicDataset(OmniDatasetForSequenceClassification):
        def __init__(self, data_path, tokenizer, **kwargs):
            super().__init__(data_path, tokenizer, **kwargs)
            # Custom initialization logic
        
        def _load_data(self, data_path):
            # Custom data loading logic
            data = self._load_json(data_path)
            return self._process_data(data)
        
        def _process_data(self, data):
            # Custom data processing
            processed_data = []
            for item in data:
                # Custom processing logic
                processed_item = self._process_item(item)
                processed_data.append(processed_item)
            return processed_data

Tokenizer Implementation
~~~~~~~~~~~~~~~~~~~~~~~~

Custom tokenizers should inherit from the abstract tokenizer:

.. code-block:: python

    from omnigenbench import OmniTokenizer
    
    class CustomGenomicTokenizer(OmniTokenizer):
        def __init__(self, base_tokenizer, **kwargs):
            super().__init__(base_tokenizer, **kwargs)
            # Custom initialization
        
        def tokenize(self, sequence, **kwargs):
            # Custom tokenization logic
            tokens = self._custom_tokenize(sequence)
            return [tokens]
        
        def _custom_tokenize(self, sequence):
            # Implement custom tokenization strategy
            # Example: k-mer tokenization
            k = self.k if hasattr(self, 'k') else 3
            tokens = []
            for i in range(len(sequence) - k + 1):
                tokens.append(sequence[i:i+k])
            return tokens

Metric Implementation
~~~~~~~~~~~~~~~~~~~~~

Custom metrics should follow the abstract metric pattern:

.. code-block:: python

    from omnigenbench import OmniMetric
    from sklearn.metrics import custom_metric
    
    class CustomGenomicMetric(OmniMetric):
        def __init__(self, ignore_y=None, **kwargs):
            super().__init__(ignore_y=ignore_y, **kwargs)
            self.metric_name = "custom_metric"
        
        def compute_metric(self, y_true, y_pred, **kwargs):
            # Filter out ignored labels
            mask = y_true != self.ignore_y
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            
            # Compute custom metric
            score = custom_metric(y_true_filtered, y_pred_filtered)
            
            return {self.metric_name: score}

Best Practices
--------------

1. **Inheritance**: Always inherit from the appropriate abstract base class
2. **Method Implementation**: Implement all required abstract methods
3. **Error Handling**: Provide meaningful error messages for invalid inputs
4. **Documentation**: Include comprehensive docstrings with examples
5. **Testing**: Write unit tests for all custom implementations
6. **Consistency**: Follow the established patterns and conventions

Common Patterns
---------------

Model-Data Integration
~~~~~~~~~~~~~~~~~~~~~~

The abstract classes are designed to work together seamlessly:

.. code-block:: python

    # Initialize components
    tokenizer = OmniSingleNucleotideTokenizer.from_pretrained("model_name")
    model = OmniModelForSequenceClassification("model_path", tokenizer)
    dataset = OmniDatasetForSequenceClassification("data.json", tokenizer)
    metric = ClassificationMetric()
    
    # Training loop
    for batch in dataset:
        outputs = model(**batch)
        loss = outputs.loss
        # Backward pass and optimization
    
    # Evaluation
    predictions = model.predict(test_sequences)
    results = metric.compute_metric(y_true, predictions['predictions'])

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

All components support flexible configuration:

.. code-block:: python

    # Model configuration
    model_config = {
        'max_length': 512,
        'num_labels': 2,
        'dropout': 0.1
    }
    
    # Dataset configuration
    dataset_config = {
        'max_length': 512,
    }
    
    # Tokenizer configuration
    tokenizer_config = {
        'convert_u_to_t': True,
        'add_whitespace': False,
        'lowercase': False
    }
    
    # Metric configuration
    metric_config = {
        'ignore_y': -100,
        'average': 'weighted'
    }

Error Handling
~~~~~~~~~~~~~~

Robust error handling is built into the abstract classes:

.. code-block:: python

    try:
        model = OmniModelForSequenceClassification("invalid_path", tokenizer)
    except FileNotFoundError:
        print("Model not found, please check the path")
    
    try:
        dataset = OmniDatasetForSequenceClassification("invalid_data.json", tokenizer)
    except ValueError as e:
        print(f"Invalid data format: {e}")
    
    try:
        metric = ClassificationMetric()
        results = metric.compute_metric(y_true, y_pred)
    except ValueError as e:
        print(f"Invalid inputs for metric computation: {e}")

Performance Considerations
--------------------------

1. **Memory Efficiency**: Use appropriate data types and batch sizes
2. **Caching**: Implement caching for expensive operations
3. **Parallelization**: Use multi-processing for data loading when possible
4. **GPU Utilization**: Ensure proper GPU memory management
5. **Profiling**: Monitor performance bottlenecks and optimize accordingly

Extension Points
----------------

The abstract classes provide several extension points for customization:

1. **Custom Loss Functions**: Override loss computation methods
2. **Custom Preprocessing**: Implement custom data preprocessing pipelines
3. **Custom Metrics**: Add new evaluation metrics
4. **Custom Tokenization**: Implement new tokenization strategies
5. **Custom Data Formats**: Add support for new data formats

This modular design allows for easy extension while maintaining consistency across the framework.
