# OmniGenBench 框架架构


## 1. 面向Config的架构详细图

```mermaid
classDiagram
    class AutoConfig {
        <<Core Config Class>>
        +model_config: Dict
        +data_config: Dict
        +training_config: Dict
        +metric_config: Dict
        +tokenizer_config: Dict
        +hyperparameters: Dict
        +load_config(path: str)
        +save_config(path: str)
        +validate_config()
    }

    class OmniModel {
        <<Abstract Model Base>>
        +config: AutoConfig
        +model_name: str
        +model_type: str
        +forward(inputs)
        +predict(inputs)
        +inference(inputs)
        +save(path: str)
    }

    class OmniDataset {
        <<Abstract Dataset Base>>
        +config: AutoConfig
        +data_path: str
        +data_format: str
        +load_from_source
        +prepare_input()
        +split_data()
    }

    class OmniTokenizer {
        <<Abstract Tokenizer Base>>
        +config: AutoConfig
        +vocab_size: int
        +tokenizer_type: str
        +encode(sequence: str)
        +decode(tokens: List)
        +tokenize(sequence: str)
    }

    class OmniMetric {
        <<Abstract Metric Base>>
        +config: AutoConfig
        +metric_names: List
        +compute(predictions, labels)
        +aggregate(results)
        +visualize(results)
    }

    class BaseTrainer {
        <<Abstract Trainer Base>>
        +config: AutoConfig
        +model: OmniModel
        +dataset: OmniDataset
        +metric: OmniMetric
        +train()
        +evaluate()
        +test()
        +setup_training()
        +save_model(path: str)
    }

    class ModelHub {
        <<Model Hub>>
        +model_registry: Dict
        +load_model(model_name: str)
        +save_model(model: OmniModel, name: str)
    }

    class HuggingFaceHub {
        <<External Service>>
        +upload_model(model_path: str, repo_id: str)
        +download_model(repo_id: str, local_path: str)
        +upload_dataset(dataset_path: str, repo_id: str)
        +download_dataset(repo_id: str, local_path: str)
    }

    class Trainer {
        <<Concrete Trainer>>
        +optimizer: Optimizer
        +lr_scheduler: LRScheduler
        +custom_train_step()
    }

    class AccelerateTrainer {
        <<Accelerate Trainer>>
        +accelerator: Accelerator
        +distributed_train()
    }

    class HFTrainer {
        <<HF Trainer>>
        +training_args: TrainingArguments
        +compute_loss()
    }

    %% Relationships
    AutoConfig o-- OmniModel : configures
    AutoConfig o-- OmniDataset : configures
    AutoConfig o-- OmniTokenizer : configures
    AutoConfig o-- OmniMetric : configures
    AutoConfig o-- BaseTrainer : configures

    BaseTrainer *-- OmniModel : contains
    BaseTrainer *-- OmniDataset : contains
    BaseTrainer *-- OmniMetric : contains

    OmniDataset ..> OmniTokenizer : uses

    ModelHub ..> HuggingFaceHub : integrates
    OmniDataset ..> HuggingFaceHub : uploads/downloads
    OmniModel ..> HuggingFaceHub : saves/loads

    BaseTrainer <|-- Trainer
    BaseTrainer <|-- AccelerateTrainer
    BaseTrainer <|-- HFTrainer

```

## 2. 面向API模块抽象层次结构

```mermaid
classDiagram
    %% 顶级抽象基类
    class AbstractBase {
        <<abstract>>
        +config: AutoConfig
        +save_config()
        +validate()
    }
    
    %% 核心抽象基类
    class OmniModel {
        <<abstract>>
        +forward(inputs)
        +predict(inputs)
        +inference(inputs)
        +train_step(batch)
        +eval_step(batch)
    }
    
    class OmniDataset {
        <<abstract>>
        +load_data_from_source()
        +prepare_input()
        +collate_fn(batch)
    }
    
    class OmniTokenizer {
        <<abstract>>
        +encode(sequence)
        +decode(tokens)
        +tokenize(sequence)
        +build_vocab()
    }
    
    class OmniMetric {
        <<abstract>>
        +compute(predictions, labels)
        +aggregate(results)
    }
    
    class BaseTrainer {
        <<abstract>>
        +train()
        +evaluate()
        +test()
        +setup_training()
        +train_epoch()
        +eval_epoch()
    }
    
    %% 具体实现类
    class ClassificationModel {
        +num_classes: int
        +classify(inputs)
        +output_attnentions()
        +get_logits()
        +compute_loss()
    }
    
    class RegressionModel {
        +output_dim: int
        +regress(inputs)
        +output_attentions()
        +get_logits()
        +compute_loss()
    }
    
    class SequenceDataset {
        +OmniDataset
        +sequences: List
        +labels: List
    }
    
    class BPETokenizer {
        +vocab_file: str
        +bpe_encode()
        +bpe_decode()
        +learn_bpe()
    }
    
    class SNTokenizer {
        +nucleotides: List
        +snt_encode()
        +snt_decode()
    }
    
    class ClassificationMetric {
        +accuracy()
        +f1_score()
        +precision_recall()
    }
    
    class RegressionMetric {
        +mse()
        +mae()
        +r2_score()
    }
    
    class Trainer {
        +device: str
        +native_train()
        +mixed_precision()
    }
    
    class AccelerateTrainer {
        +accelerator: Accelerator
        +distributed_train()
        +multi_gpu_support()
    }
    
    class HFTrainer {
        +training_args: TrainingArguments
        +hf_integration()
        +transformers_support()
    }
    
    %% 继承关系
    AbstractBase <|-- OmniModel
    AbstractBase <|-- OmniDataset
    AbstractBase <|-- OmniTokenizer
    AbstractBase <|-- OmniMetric
    AbstractBase <|-- BaseTrainer
    
    %% 模型继承
    OmniModel <|-- ClassificationModel
    OmniModel <|-- RegressionModel
    
    %% 数据继承
    OmniDataset <|-- SequenceDataset
    
    %% 分词器继承
    OmniTokenizer <|-- BPETokenizer
    OmniTokenizer <|-- DNATokenizer
    
    %% 指标继承
    OmniMetric <|-- ClassificationMetric
    OmniMetric <|-- RegressionMetric
    
    %% 训练器继承
    BaseTrainer <|-- Trainer
    BaseTrainer <|-- AccelerateTrainer
    BaseTrainer <|-- HFTrainer
```


## 3. 面向Interface框架架构

```mermaid
graph TB
    %% 样式定义
    classDef userLayer fill:#e3f2fd,stroke:#1565c0,stroke-width:3px,color:#000
    classDef cliLayer fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
    classDef apiLayer fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000
    classDef configLayer fill:#fce4ec,stroke:#c2185b,stroke-width:3px,color:#000
    classDef hubLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,color:#000
    
    %% 用户结构层
    subgraph "👥 用户结构层"
        CLI["🖥️ 命令行接口<br/>Command Line Interface"]
        API["🐍 Python API<br/>Programming Interface"]
    end
    
    %% 命令行接口层
    subgraph "⚡ 命令行接口层"
        AutoTrain["AutoTrain<br/>自动训练命令<br/>autotrain"]
        AutoBench["AutoBench<br/>自动基准测试<br/>autobench"]
        CLIConfig["CLI配置<br/>--config --model --dataset"]
    end
    
    %% API模块层
    subgraph "🧩 API模块层"
        subgraph "📊 数据模块"
            OmniDataset["OmniDataset<br/>抽象数据基类<br/>+load_from_source()<br/>+prepare_input()"]
            OmniTokenizer["OmniTokenizer<br/>抽象分词器基类<br/>+encode()<br/>+decode()<br/>+tokenize()"]
        end
        
        subgraph "🧠 模型模块"
            OmniModel["OmniModel<br/>抽象模型基类<br/>+forward()<br/>+predict()<br/>+inference()"]
            ModelHub["ModelHub<br/>模型中心<br/>+load_model()<br/>+save_model()<br/>+push_to_hub()"]
        end
        
        subgraph "📏 指标模块"
            OmniMetric["OmniMetric<br/>抽象指标基类<br/>+compute()<br/>+aggregate()<br/>+visualize()"]
        end
        
        subgraph "🚀 训练模块"
            BaseTrainer["BaseTrainer<br/>抽象训练器基类<br/>+train()<br/>+evaluate()<br/>+test()<br/>+setup_training()"]
            Trainer["Trainer<br/>原生训练器<br/>+native_train()<br/>+mixed_precision()"]
            AccelerateTrainer["AccelerateTrainer<br/>分布式训练器<br/>+distributed_train()<br/>+multi_gpu_support()"]
            HFTrainer["HFTrainer<br/>HF集成训练器<br/>+hf_integration()<br/>+transformers_support()"]
        end
    end
    
    %% 配置核心层
    subgraph "⚙️ 配置核心"
        AutoConfig["AutoConfig<br/>统一配置管理<br/>+model_config: Dict<br/>+data_config: Dict<br/>+training_config: Dict<br/>+metric_config: Dict<br/>+tokenizer_config: Dict<br/>+load_config()<br/>+save_config()<br/>+validate_config()"]
    end
    
    %% HuggingFace Hub集成
    subgraph "🤗 HuggingFace Hub 集成"
        HFHub["HuggingFace Hub<br/>外部服务集成<br/>📦 数据集存储/读取<br/>🧠 模型存储/读取<br/>📊 实验跟踪<br/>🔄 版本控制"]
    end
    
    %% 连接关系
    CLI --> AutoTrain
    CLI --> AutoBench
    CLI --> CLIConfig
    
    API --> OmniDataset
    API --> OmniTokenizer
    API --> OmniModel
    API --> ModelHub
    API --> OmniMetric
    API --> BaseTrainer
    
    AutoTrain --> AutoConfig
    AutoBench --> AutoConfig
    CLIConfig --> AutoConfig
    
    AutoConfig --> OmniDataset
    AutoConfig --> OmniTokenizer
    AutoConfig --> OmniModel
    AutoConfig --> OmniMetric
    AutoConfig --> BaseTrainer
    
    BaseTrainer --> Trainer
    BaseTrainer --> AccelerateTrainer
    BaseTrainer --> HFTrainer
    
    ModelHub <--> HFHub
    OmniDataset <--> HFHub
    
    %% 应用样式
    class CLI,API userLayer
    class AutoTrain,AutoBench,CLIConfig cliLayer
    class OmniDataset,OmniTokenizer,OmniModel,ModelHub,OmniMetric,BaseTrainer,Trainer,AccelerateTrainer,HFTrainer apiLayer
    class AutoConfig configLayer
    class HFHub hubLayer
```

## 4. 微调流程图

```mermaid
flowchart TB
    %% 样式
    classDef configStep fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#000
    classDef processStep fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#000
    classDef dataStep fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef deployStep fill:#fff8e1,stroke:#f57c00,stroke-width:2px,color:#000
    classDef autoStep fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000

    Start(["选择序列预测微调任务"])

    %% 选择流程方式
    Mode{{选择微调流程?}}
    Start --> Mode

    %% 手动流程
    subgraph Manual["手动流程"]
    direction TB
      M1["配置模型与分词器"]:::configStep
      M2["读取/分词/校验数据（prepare_input）"]:::processStep
      M3["设置训练超参"]:::processStep
      M4["选择/实现评估指标"]:::processStep
      M5["实例化 Trainer（Trainer/Accelerate/HF）"]:::processStep
      M6["训练 & 验证（train/evaluate）"]:::processStep
      MQ{{指标是否达标?}}:::processStep
      M7["保存模型与指标报告"]:::dataStep
      M8["分享模型到社区（ModelHub/HF Hub）"]:::dataStep
      M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> MQ
      MQ -- 否 --> M3
      MQ -- 是 --> M7 --> M8
    end

    %% AutoTrain
    subgraph AutoTrain["基于配置的自动训练"]
    direction TB
      A1["创建 AutoConfig<br/>（将前置步骤写入Config）"]:::configStep
      A2["AutoTrain<br/>autotrain --config config.py"]:::autoStep
      AQ{{指标是否达标?}}:::processStep
      A3["自动评估与保存"]:::dataStep
      A4["上传模型到Hub（ModelHub/HF Hub）"]:::dataStep
      A1 --> A2 --> AQ
      AQ -- 否 --> A1
      AQ -- 是 --> A3 --> A4
    end

    %% 合流到部署
    Mode -- 手动 --> M1
    Mode -- Auto --> A1

    subgraph Deploy["模型部署实践"]
    direction TB
      D1["加载模型 & 推理接口（inference）"]:::deployStep
      D2["FastAPI Serving"]:::deployStep
      Feed{{测试通过?}}:::deployStep
    end

    M8 --> D1
    A4 --> D1
    D1 --> D2 --> Feed
    Feed -- 是 --> M2
    Feed -- 否 --> A1
    Feed -- 是 --> End(["序列预测"]):::deployStep

```


## 框架设计原则实现

### 1. ✅ 用户结构层分离
- **命令行接口**: AutoTrain、AutoBench等CLI命令
- **Python API**: 数据集、模型、分词器、指标库、训练器等模块

### 2. ✅ 抽象基类集成
- **OmniDataset**: 数据集抽象基类
- **OmniModel**: 模型抽象基类  
- **OmniTokenizer**: 分词器抽象基类
- **OmniMetric**: 指标库抽象基类
- **BaseTrainer**: 训练器抽象基类

### 3. ✅ 主要类方法和属性
每个模块都详细列出了核心方法和属性，包括：
- 数据处理方法 (`__getitem__`, `preprocess`)
- 模型推理方法 (`forward`, `predict`, `inference`)
- 配置加载方法 (`load_from_config`)

### 4. ✅ 配置驱动框架
- **AutoConfig**: 统一配置管理所有模块信息和超参数
- **配置注入**: 所有模块都通过config进行初始化和配置

### 5. ✅ 完整微调流程
详细的9步微调流程：
```
加载Config → 设置模型分词器 → 读取数据分词 → 设置Metric → 
实例化训练器 → AutoTrain引擎 → 保存ModelHub → 读取模型 → 部署
```

### 6. ✅ HuggingFace Hub集成
- 支持模型和数据集的上传下载
- 版本控制和实验���踪
- 与社区生态深度集成
