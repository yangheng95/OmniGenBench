# 翻译效率预测 - BPP特征融合模型

## 📋 概述

本模块实现了一个基于 **OmniGenBench** 框架的翻译效率（Translation Efficiency, TE）预测模型，通过融合 **RNA序列特征** 和 **碱基配对概率（Base Pairing Probability, BPP）** 结构特征来提高预测准确性。

### 🎯 核心创新

1. **结构特征融合**: 将RNA二级结构信息（BPP矩阵）与序列embeddings结合
2. **端到端训练**: 联合优化所有组件，无需预训练
3. **模块化设计**: 易于替换基础模型或修改特征处理器
4. **生产就绪**: 完整的训练、评估和推理pipeline

---

## 🏗️ 模型架构

```
输入: RNA序列
    │
    ├─────────────────────┬──────────────────────┐
    │                     │                      │
    ▼                     ▼                      ▼
[Tokenizer]        [ViennaRNA]           [Sequence Info]
    │                     │                      │
    ▼                     ▼                      │
[Transformer]      [BPP Matrix]                 │
    │               512×512                      │
    │                     │                      │
    ▼                     ▼                      │
[CLS Token]        [CNN Processor]               │
 Embedding          3-Layer CNN                  │
  (768-d)          + Global Pool                 │
    │                  (128-d)                   │
    │                     │                      │
    └──────────┬──────────┘                      │
               │                                 │
               ▼                                 │
         [Feature Fusion]                        │
         Gated MLP (256-d)                       │
               │                                 │
               ▼                                 │
         [Classifier]                            │
           2 classes                             │
               │                                 │
               ▼                                 │
    Prediction: High TE / Low TE                 │
```

### 关键组件

#### 1. **TEDatasetWithBPP** - 自定义数据集类
- **功能**: 在数据加载时计算BPP矩阵
- **特点**:
  - 自动序列填充/截断到512长度
  - 使用ViennaRNA配分函数计算BPP
  - BPP结果缓存（提速）
  - 优雅的错误处理

```python
class TEDatasetWithBPP(OmniDatasetForSequenceClassification):
    def compute_bpp_matrix(self, sequence: str) -> np.ndarray:
        """使用ViennaRNA计算碱基配对概率矩阵"""
        fc = RNA.fold_compound(sequence)
        fc.pf()  # 计算配分函数
        bpp = fc.bpp()  # 获取BPP矩阵
        # 转换为对称numpy数组
        return bpp_matrix
    
    def prepare_input(self, instance, **kwargs):
        """准备输入: 序列tokenization + BPP计算"""
        # 1. 提取序列并转换T->U
        # 2. 填充/截断到512
        # 3. 计算BPP矩阵
        # 4. Tokenize序列
        # 5. 返回所有特征
```

#### 2. **BPPProcessor** - BPP特征提取器
- **架构**: 3层CNN + 全局平均池化
- **输入**: `[batch, 1, 512, 512]` BPP矩阵
- **输出**: `[batch, 128]` 结构特征向量

```python
self.cnn = nn.Sequential(
    # Layer 1: 512×512 -> 256×256
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    
    # Layer 2: 256×256 -> 128×128
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    
    # Layer 3: 128×128 -> 1×1 (global pool)
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
)
```

#### 3. **FeatureFusion** - 门控特征融合模块
- **机制**: 自适应门控融合（Gated Fusion）
- **公式**: `output = gate * seq_features + (1 - gate) * bpp_features`
- **优势**: 模型自动学习两种特征的权重

```python
class FeatureFusion(nn.Module):
    def forward(self, seq_features, bpp_features):
        # 投影到共同空间
        seq_proj = self.seq_proj(seq_features)
        bpp_proj = self.bpp_proj(bpp_features)
        
        # 计算门控权重
        gate = self.gate(torch.cat([seq_features, bpp_features], dim=1))
        
        # 门控融合
        fused = gate * seq_proj + (1 - gate) * bpp_proj
        
        # MLP处理
        return self.fusion_mlp(fused)
```

#### 4. **TEModelWithBPP** - 完整模型
- **继承**: `OmniModelForSequenceClassification`
- **兼容性**: 支持所有OmniGenBench基础模型
- **特点**: 端到端可微分训练

---

## 🚀 快速开始

### 安装依赖

```bash
# 安装OmniGenBench
pip install omnigenbench -U

# 安装ViennaRNA (必需)
conda install -c bioconda viennarna

# 或者使用pip (Linux/Mac)
pip install ViennaRNA
```

### 方法1: 一键训练（推荐）

```python
from te_bpp_model import train_te_model

# 使用你的数据训练
results = train_te_model(
    model_name="yangheng/PlantRNA-FM",  # 或其他基础模型
    train_file="data/train.json",
    valid_file="data/valid.json",
    test_file="data/test.json",
    max_length=512,
    batch_size=16,
    epochs=30,
    learning_rate=2e-5,
    output_dir="my_te_model"
)

print(f"Test F1 Score: {results['f1_score']:.4f}")
```

### 方法2: 使用演示数据测试

```python
from te_bpp_model import train_te_model

# 自动创建演示数据集并训练
results = train_te_model(
    model_name="yangheng/PlantRNA-FM",
    use_demo=True,  # 使用合成数据
    batch_size=8,
    epochs=10,
    output_dir="te_demo_model"
)
```

### 方法3: 自定义训练流程

```python
from te_bpp_model import TEModelWithBPP, TEDatasetWithBPP
from omnigenbench import OmniTokenizer, AccelerateTrainer, ClassificationMetric

# 1. 初始化
tokenizer = OmniTokenizer.from_pretrained("yangheng/PlantRNA-FM")
label2id = {"0": 0, "1": 1}

# 2. 加载数据集
train_dataset = TEDatasetWithBPP(
    "train.json",
    tokenizer=tokenizer,
    max_length=512,
    label2id=label2id
)
valid_dataset = TEDatasetWithBPP("valid.json", tokenizer, 512, label2id=label2id)
test_dataset = TEDatasetWithBPP("test.json", tokenizer, 512, label2id=label2id)

# 3. 创建模型
model = TEModelWithBPP(
    config_or_model="yangheng/PlantRNA-FM",
    tokenizer=tokenizer,
    num_labels=2,
    label2id=label2id
)

# 4. 训练
trainer = AccelerateTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    test_dataset=test_dataset,
    compute_metrics=[ClassificationMetric().f1_score],
    batch_size=8,
    num_train_epochs=20,
    learning_rate=2e-5
)

metrics = trainer.train()
```

---

## 📊 数据格式

### 输入格式: JSON Lines

每行一个JSON对象，包含`sequence`和`label`字段：

```json
{"sequence": "AUGCAUGCAUGCGCGCGCGC...", "label": 1}
{"sequence": "UUUAAAGGGCCCUUUAAAGGG...", "label": 0}
```

- **sequence**: RNA序列（A, U, G, C, N）
  - 支持T/U自动转换
  - 自动填充到512长度（短序列用'N'填充）
  - 自动截断到512长度（长序列）
  
- **label**: 
  - `0`: 低翻译效率（Low TE）
  - `1`: 高翻译效率（High TE）

### 数据集结构

```
translation_efficiency_prediction/
├── train.json      # 训练集 (推荐: 1000+ samples)
├── valid.json      # 验证集 (推荐: 200+ samples)
└── test.json       # 测试集 (推荐: 200+ samples)
```

### 创建你自己的数据集

```python
import json

# 准备数据
data = [
    {"sequence": "AUGCAUGC...", "label": 1},
    {"sequence": "UUUAAAGGG...", "label": 0},
    # ... 更多样本
]

# 保存为JSON Lines格式
with open("train.json", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
```

---

## 🔧 模型推理

### 单序列预测

```python
from omnigenbench import ModelHub

# 加载训练好的模型
model = ModelHub.load("my_te_model")

# 预测
sequence = "AUGCAUGCAUGCGCGCGCGC" * 20
outputs = model.inference({"sequence": sequence})

prediction = outputs["predictions"][0]  # 0 or 1
confidence = outputs["confidence"]       # 0.0 - 1.0

print(f"Prediction: {'High TE' if prediction == 1 else 'Low TE'}")
print(f"Confidence: {confidence:.4f}")
```

### 批量预测

```python
sequences = [
    "AUGCAUGCAUGC...",
    "UUUAAAGGGCCC...",
    "GCGCGCGCGCGC..."
]

for seq in sequences:
    outputs = model.inference({"sequence": seq})
    print(f"Sequence: {seq[:30]}...")
    print(f"  -> Prediction: {outputs['predictions'][0]}")
    print(f"  -> Confidence: {outputs['confidence']:.4f}\n")
```

### 提取特征用于下游分析

```python
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        bpp_matrix=bpp_matrix
    )
    
    sequence_features = outputs["sequence_features"]  # [batch, 768]
    bpp_features = outputs["bpp_features"]            # [batch, 128]
    logits = outputs["logits"]                        # [batch, 2]
```

---

## 🎨 高级定制

### 1. 更换基础模型

```python
# 使用其他植物基因组模型
model = TEModelWithBPP(
    config_or_model="yangheng/OmniGenome-186M",  # 通用基因组模型
    tokenizer=tokenizer,
    num_labels=2
)

# 使用DNABERT
model = TEModelWithBPP(
    config_or_model="zhihan1996/DNA_bert_6",
    tokenizer=tokenizer,
    num_labels=2
)
```

### 2. 调整BPP处理器

```python
# 使用更深的CNN
class DeepBPPProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # 添加更多层...
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    
    def forward(self, bpp_matrix):
        return self.cnn(bpp_matrix).flatten(1)

# 替换默认处理器
model.bpp_processor = DeepBPPProcessor()
```

### 3. 修改融合策略

```python
# 简单拼接（不使用门控）
class SimpleFusion(nn.Module):
    def __init__(self, seq_dim, bpp_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(seq_dim + bpp_dim, 512),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )
    
    def forward(self, seq_feat, bpp_feat):
        return self.mlp(torch.cat([seq_feat, bpp_feat], dim=1))

# 使用
model.fusion = SimpleFusion(768, 128, 256)
```

### 4. 多任务学习

```python
class MultiTaskTEModel(TEModelWithBPP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加辅助任务head（例如：预测表达量）
        self.expression_head = nn.Linear(256, 1)
    
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        # 添加辅助任务输出
        fused_features = self.fusion(
            outputs["sequence_features"],
            outputs["bpp_features"]
        )
        outputs["expression_pred"] = self.expression_head(fused_features)
        return outputs
```

---

## 📈 性能优化建议

### 训练加速

```python
# 1. 使用混合精度训练
trainer = AccelerateTrainer(
    model=model,
    train_dataset=train_dataset,
    mixed_precision="fp16",  # 或 "bf16" (A100+)
    batch_size=16,           # 增大batch size
)

# 2. 使用梯度累积（模拟大batch）
trainer = AccelerateTrainer(
    model=model,
    train_dataset=train_dataset,
    batch_size=4,
    gradient_accumulation_steps=4,  # 有效batch=16
)

# 3. 多GPU训练
# 自动检测并使用所有可用GPU
trainer = AccelerateTrainer(
    model=model,
    train_dataset=train_dataset,
)
```

### BPP计算优化

```python
# 使用并行计算BPP（适合大数据集预处理）
from multiprocessing import Pool

def precompute_bpp_matrices(sequences, num_workers=8):
    """预计算所有BPP矩阵并保存"""
    with Pool(num_workers) as pool:
        bpp_matrices = pool.map(compute_bpp_matrix, sequences)
    return bpp_matrices

# 保存预计算的BPP
import pickle
with open("bpp_cache.pkl", "wb") as f:
    pickle.dump(bpp_matrices, f)
```

### 内存优化

```python
# 对于长序列数据集
dataset = TEDatasetWithBPP(
    "train.json",
    tokenizer=tokenizer,
    max_length=512,  # 减少max_length
    label2id=label2id
)

# 清理BPP缓存
dataset._bpp_cache.clear()
```

---

## 🧪 实验建议

### 超参数搜索

```python
# 推荐的超参数范围
hyperparams = {
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "batch_size": [8, 16, 32],
    "bpp_dim": [64, 128, 256],
    "fusion_dim": [128, 256, 512],
}

# 使用网格搜索或随机搜索
best_f1 = 0
best_config = None

for lr in hyperparams["learning_rate"]:
    for bs in hyperparams["batch_size"]:
        model = TEModelWithBPP(...)
        results = train_te_model(
            learning_rate=lr,
            batch_size=bs,
            epochs=20
        )
        if results["f1_score"] > best_f1:
            best_f1 = results["f1_score"]
            best_config = {"lr": lr, "batch_size": bs}
```

### 消融实验

```python
# 1. 仅使用序列特征（baseline）
class SequenceOnlyModel(OmniModelForSequenceClassification):
    def forward(self, input_ids, attention_mask, bpp_matrix=None, labels=None):
        # 忽略bpp_matrix
        outputs = self.model(input_ids, attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        # ... 计算loss
        return {"loss": loss, "logits": logits}

# 2. 仅使用BPP特征
class BPPOnlyModel(nn.Module):
    def forward(self, bpp_matrix, labels=None):
        bpp_features = self.bpp_processor(bpp_matrix)
        logits = self.classifier(bpp_features)
        # ... 计算loss
        return {"loss": loss, "logits": logits}

# 3. 完整模型（序列 + BPP）
# 使用 TEModelWithBPP

# 比较三者性能
```

---

## 📚 技术细节

### BPP矩阵详解

**定义**: BPP[i][j] 表示位置i和j之间形成碱基配对的概率

**计算方法**: 使用ViennaRNA的配分函数算法
```python
fc = RNA.fold_compound(sequence)
fc.pf()  # 计算Boltzmann配分函数 Z = Σ_s exp(-E(s)/RT)
bpp = fc.bpp()  # BPP[i][j] = P(i-j paired) = Σ_s δ_{ij}(s) exp(-E(s)/RT) / Z
```

**性质**:
- 对称矩阵: BPP[i][j] = BPP[j][i]
- 值域: [0, 1]
- 对角线为0（自身不能配对）
- 高值表示该配对在热力学上稳定

**生物学意义**:
- 反映RNA二级结构的灵活性
- 与翻译效率相关（结构影响核糖体结合和移动）
- 捕获5' UTR和CDS起始区的结构特征

### 模型容量分析

```python
from te_bpp_model import TEModelWithBPP
from omnigenbench import OmniTokenizer

tokenizer = OmniTokenizer.from_pretrained("yangheng/PlantRNA-FM")
model = TEModelWithBPP("yangheng/PlantRNA-FM", tokenizer, num_labels=2)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 预期输出 (PlantRNA-FM)
# Total parameters: ~35,000,000
# Trainable parameters: ~35,000,000
```

**组件参数分解**:
- PlantRNA-FM base: ~35M
- BPP Processor (CNN): ~200K
- Feature Fusion: ~500K
- Classifier: ~512

### 计算复杂度

**训练时间复杂度**:
- BPP计算: O(N³) per sequence (ViennaRNA)
- Transformer: O(N²·D) per forward pass
- CNN: O(N²·K²·C) (K=kernel size, C=channels)
- Total: O(N³ + N²·D) dominated by BPP

**推理时间** (512长度序列, single GPU):
- BPP计算: ~0.5-1s
- Model forward: ~0.01s
- Total: ~0.5-1s per sequence

**内存使用** (512长度, batch=8):
- BPP matrices: 8 × 512² × 4 bytes = ~8MB
- Model activations: ~2GB
- Total: ~2-3GB GPU memory

---

## 🐛 故障排查

### 常见问题

#### 1. ViennaRNA安装失败

**症状**: `ImportError: No module named 'RNA'`

**解决方案**:
```bash
# macOS/Linux
conda install -c bioconda viennarna

# Windows (需要WSL)
# 1. 安装WSL: https://docs.microsoft.com/en-us/windows/wsl/install
# 2. 在WSL中安装conda
# 3. conda install -c bioconda viennarna
```

#### 2. BPP计算失败

**症状**: `Warning: Failed to compute BPP`

**原因**: 序列包含无效字符或太短

**解决方案**:
```python
# 检查序列
sequence = sequence.upper().replace("T", "U")
valid_chars = set("AUGCN")
if not all(c in valid_chars for c in sequence):
    print(f"Invalid characters in sequence")

# 过滤无效序列
filtered_data = [
    item for item in data 
    if len(item["sequence"]) >= 50 and 
       all(c in "AUGCTN" for c in item["sequence"].upper())
]
```

#### 3. CUDA内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 方案1: 减小batch size
trainer = AccelerateTrainer(model=model, batch_size=4)

# 方案2: 减小序列长度
dataset = TEDatasetWithBPP(..., max_length=256)

# 方案3: 使用梯度检查点
model.model.gradient_checkpointing_enable()

# 方案4: 使用CPU（速度慢）
device = torch.device("cpu")
model = model.to(device)
```

#### 4. 训练不收敛

**症状**: Loss不下降或波动

**诊断**:
```python
# 检查数据平衡
from collections import Counter
labels = [item["label"] for item in train_data]
print(Counter(labels))  # 应该接近1:1

# 检查学习率
# 尝试降低学习率
trainer = AccelerateTrainer(learning_rate=1e-5)

# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

---

## 📖 参考资料

### 相关论文

1. **ViennaRNA Package**
   - Lorenz, R. et al. (2011). ViennaRNA Package 2.0. Algorithms Mol Biol, 6, 26.
   - [Link](https://almob.biomedcentral.com/articles/10.1186/1748-7188-6-26)

2. **Translation Efficiency Prediction**
   - Sample, P. J. et al. (2019). Human 5' UTR design and variant effect prediction from a massively parallel translation assay. Nat Biotechnol, 37(7), 803-809.

3. **RNA Structure and Translation**
   - Bevilacqua, P. C. et al. (2016). Structures mRNA: A New Frontier in RNA Biology and Drug Targets. Nat Rev Mol Cell Biol, 17(7), 436-451.

### OmniGenBench文档

- **主页**: [https://github.com/yangheng95/OmniGenBench](https://github.com/yangheng95/OmniGenBench)
- **文档**: [https://omnigenbench.readthedocs.io/](https://omnigenbench.readthedocs.io/)
- **示例**: [https://github.com/yangheng95/OmniGenBench/tree/master/examples](https://github.com/yangheng95/OmniGenBench/tree/master/examples)

### 基础模型

- **PlantRNA-FM**: [yangheng/PlantRNA-FM](https://huggingface.co/yangheng/PlantRNA-FM)
- **OmniGenome**: [yangheng/OmniGenome-186M](https://huggingface.co/yangheng/OmniGenome-186M)

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 改进建议
- [ ] 支持多标签分类（同时预测TE和稳定性）
- [ ] 添加注意力可视化（显示哪些配对对预测最重要）
- [ ] 集成其他结构预测方法（RNAfold, Mfold）
- [ ] 支持可变长度序列（无需padding）
- [ ] 添加模型解释性分析工具

---

## 📄 许可证

本代码遵循 MIT License，详见 [LICENSE](../../LICENSE) 文件。

---

## 📧 联系方式

- **作者**: YANG, HENG (杨恒)
- **邮箱**: hy345@exeter.ac.uk
- **主页**: [https://yangheng95.github.io](https://yangheng95.github.io)
- **GitHub**: [@yangheng95](https://github.com/yangheng95)

---

**最后更新**: 2025年11月1日
