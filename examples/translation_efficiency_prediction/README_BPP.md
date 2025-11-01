# 翻译效率预测 - BPP 特征融合实现

> 🌟 **完整的生产级实现** - 基于OmniGenBench框架，融合RNA序列和结构特征

本目录包含基于 **OmniGenBench** 框架的翻译效率（TE）预测模型，通过融合RNA序列特征和碱基配对概率（BPP）结构特征来提高预测准确性。

## 📚 快速导航

| 文档 | 说明 | 推荐度 |
|------|------|--------|
| **[开始使用](#🚀-快速开始)** | 5分钟入门指南 | ⭐⭐⭐⭐⭐ |
| **[代码实现](./te_bpp_model.py)** | 完整的Python实现 | ⭐⭐⭐⭐⭐ |
| **[使用示例](./example_usage.py)** | 5个实用场景 | ⭐⭐⭐⭐ |
| **[测试套件](./test_te_bpp_model.py)** | 单元测试 | ⭐⭐⭐ |
| **[详细文档](./README_TE_BPP.md)** | 完整技术文档 | ⭐⭐⭐⭐ |
| **[架构说明](./ARCHITECTURE.md)** | 可视化架构图 | ⭐⭐⭐⭐ |
| **[实现总结](./IMPLEMENTATION_SUMMARY.md)** | 开发者指南 | ⭐⭐⭐ |

## 📋 项目结构

```
translation_efficiency_prediction/
│
├── 🎯 核心实现
│   └── te_bpp_model.py                 # 生产级完整实现 (800+ 行)
│       ├── TEDatasetWithBPP            # 自定义数据集类
│       ├── BPPProcessor                # CNN特征提取器  
│       ├── FeatureFusion               # 门控融合模块
│       ├── TEModelWithBPP              # 完整模型
│       └── train_te_model()            # 一键训练函数
│
├── 🧪 测试和示例
│   ├── test_te_bpp_model.py            # 测试套件 (7个测试)
│   └── example_usage.py                # 使用示例 (5个场景)
│
├── 📚 文档
│   ├── README_BPP.md                   # 本文件 - 快速入门
│   ├── README_TE_BPP.md                # 详细技术文档 (50+ 页)
│   ├── ARCHITECTURE.md                 # 架构可视化
│   └── IMPLEMENTATION_SUMMARY.md       # 实现总结
│
└── 📁 参考（旧版本）
    ├── te_with_bpp_features.py         # 初始版本
    └── quickstart_te.py                # 基础教程
```

### 🎓 学习路径

- **快速上手**: 本文档 → `example_usage.py` (示例1)
- **深入学习**: `README_TE_BPP.md` → `ARCHITECTURE.md` → 源码
- **开发调试**: `IMPLEMENTATION_SUMMARY.md` → `test_te_bpp_model.py`

## 🚀 快速开始

### 安装依赖

```bash
# 安装 OmniGenBench
pip install omnigenbench -U

# 安装 ViennaRNA (必需)
conda install -c bioconda viennarna
```

### 方法1: 一键训练（最简单）

```python
from te_bpp_model import train_te_model

# 使用你的数据
results = train_te_model(
    model_name="yangheng/PlantRNA-FM",
    train_file="data/train.json",
    valid_file="data/valid.json",
    test_file="data/test.json",
    batch_size=16,
    epochs=30,
    output_dir="my_te_model"
)
```

### 方法2: 使用示例脚本

```bash
# 查看所有示例
python example_usage.py --help

# 运行快速训练示例
python example_usage.py --example 1

# 运行批量推理示例
python example_usage.py --example 3
```

### 方法3: 运行测试

```bash
# 运行完整测试套件
python test_te_bpp_model.py
```

## 🏗️ 模型架构

```
输入: RNA序列 → [Tokenizer] → [Transformer] → 序列特征 (768-d)
                                                    ↓
              → [ViennaRNA] → BPP矩阵 → [CNN] → BPP特征 (128-d)
                                                    ↓
                        [门控融合] → 融合特征 (256-d)
                                      ↓
                              [分类器] → High TE / Low TE
```

## 🎯 核心技术点

### 1. 自定义 Dataset - 计算 BPP

```python
class TEDatasetWithBPP(OmniDatasetForSequenceClassification):
    def compute_bpp_matrix(self, sequence: str) -> np.ndarray:
        """使用ViennaRNA配分函数计算BPP"""
        fc = RNA.fold_compound(sequence)
        fc.pf()  # 计算配分函数
        bpp = fc.bpp()  # 获取BPP矩阵
        return bpp_matrix  # 对称矩阵 [seq_len, seq_len]
    
    def prepare_input(self, instance, **kwargs):
        """准备输入: 序列 + BPP矩阵"""
        # 1. 提取序列，T->U转换
        # 2. 填充/截断到512
        # 3. 计算BPP矩阵
        # 4. Tokenize序列
        # 5. 返回所有特征
        return {
            "input_ids": ...,
            "attention_mask": ...,
            "bpp_matrix": bpp_matrix,  # [512, 512]
            "labels": ...
        }
```

### 2. BPP特征处理器

```python
class BPPProcessor(nn.Module):
    """3层CNN提取BPP结构特征"""
    def __init__(self, output_dim=128):
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # 512×512
            nn.ReLU(), nn.MaxPool2d(2),       # → 256×256
            nn.Conv2d(32, 64, 3, padding=1),  # 256×256
            nn.ReLU(), nn.MaxPool2d(2),       # → 128×128
            nn.Conv2d(64, 128, 3, padding=1), # 128×128
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),     # → 1×1
        )
```

### 3. 门控特征融合

```python
class FeatureFusion(nn.Module):
    """自适应门控融合序列和BPP特征"""
    def forward(self, seq_features, bpp_features):
        # 计算门控权重
        gate = self.gate(torch.cat([seq_features, bpp_features], dim=1))
        
        # 门控融合
        fused = gate * seq_proj + (1 - gate) * bpp_proj
        
        return self.fusion_mlp(fused)  # [batch, 256]
```

### 4. 完整模型

```python
class TEModelWithBPP(OmniModelForSequenceClassification):
    """集成所有组件的完整模型"""
    def forward(self, input_ids, attention_mask, bpp_matrix, labels=None):
        # 1. Transformer处理序列
        seq_features = self.model(input_ids, attention_mask)
        
        # 2. CNN处理BPP矩阵
        bpp_features = self.bpp_processor(bpp_matrix)
        
        # 3. 门控融合
        fused = self.fusion(seq_features, bpp_features)
        
        # 4. 分类
        logits = self.classifier(fused)
        
        # 5. 计算loss
        loss = CrossEntropyLoss()(logits, labels) if labels else None
        
        return {"loss": loss, "logits": logits, ...}
```

## � 使用示例

### 示例1: 快速训练

```python
from te_bpp_model import train_te_model

results = train_te_model(
    model_name="yangheng/PlantRNA-FM",
    use_demo=True,  # 使用演示数据
    batch_size=8,
    epochs=10
)
print(f"F1 Score: {results['f1_score']:.4f}")
```

### 示例2: 模型推理

```python
from omnigenbench import ModelHub

# 加载训练好的模型
model = ModelHub.load("my_te_model")

# 预测
sequence = "AUGCAUGCGCGCGCGC" * 30
outputs = model.inference({"sequence": sequence})

print(f"Prediction: {outputs['predictions'][0]}")  # 0 or 1
print(f"Confidence: {outputs['confidence']:.4f}")
```

### 示例3: 批量预测

```python
sequences = [
    "AUGCAUGC..." * 30,
    "UUUAAAGGG..." * 30,
    "GCGCGCGC..." * 30
]

for seq in sequences:
    outputs = model.inference({"sequence": seq})
    pred = "High TE" if outputs['predictions'][0] == 1 else "Low TE"
    print(f"Sequence: {seq[:30]}... -> {pred}")
```

### 示例4: 提取特征

```python
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        bpp_matrix=bpp_matrix
    )
    
    seq_features = outputs["sequence_features"]  # [batch, 768]
    bpp_features = outputs["bpp_features"]        # [batch, 128]
    logits = outputs["logits"]                    # [batch, 2]
```

## � 数据格式

### 输入: JSON Lines格式

```json
{"sequence": "AUGCAUGCAUGCGCGCGCGC...", "label": 1}
{"sequence": "UUUAAAGGGCCCUUUAAAGGG...", "label": 0}
```

- **sequence**: RNA序列（A, U, G, C, N）
  - 自动T→U转换
  - 自动填充/截断到512长度
  
- **label**: 
  - `0`: 低翻译效率
  - `1`: 高翻译效率

### 数据集结构

```
translation_efficiency_prediction/
├── train.json      # 训练集
├── valid.json      # 验证集
└── test.json       # 测试集
```

## 🔧 高级定制

### 更换基础模型

```python
# 使用其他模型
model = TEModelWithBPP(
    config_or_model="yangheng/OmniGenome-186M",
    tokenizer=tokenizer,
    num_labels=2
)
```

### 调整BPP处理器

```python
# 使用更深的CNN
model.bpp_processor = DeepBPPProcessor(output_dim=256)
```

### 修改融合策略

```python
# 简单拼接（替代门控融合）
class SimpleFusion(nn.Module):
    def forward(self, seq_feat, bpp_feat):
        return self.mlp(torch.cat([seq_feat, bpp_feat], dim=1))

model.fusion = SimpleFusion(768, 128, 256)
```

## 💡 技术亮点

1. **模块化设计**: 每个组件独立可替换
2. **端到端训练**: 联合优化所有参数
3. **生产就绪**: 完整的训练、评估、推理pipeline
4. **高度兼容**: 支持所有OmniGenBench基础模型
5. **灵活扩展**: 易于添加新特征或修改架构

## � 故障排查

### ViennaRNA安装失败

```bash
# macOS/Linux
conda install -c bioconda viennarna

# Windows (需要WSL)
# 在WSL中安装
```

### CUDA内存不足

```python
# 减小batch size
trainer = AccelerateTrainer(batch_size=4)

# 或减小序列长度
dataset = TEDatasetWithBPP(..., max_length=256)
```

### BPP计算失败

```python
# 检查序列有效性
sequence = sequence.upper().replace("T", "U")
if not all(c in "AUGCN" for c in sequence):
    print("Invalid sequence")
```

## 📖 参考文档

- **完整技术文档**: [README_TE_BPP.md](./README_TE_BPP.md)
- **OmniGenBench**: [https://github.com/yangheng95/OmniGenBench](https://github.com/yangheng95/OmniGenBench)
- **ViennaRNA**: [https://www.tbi.univie.ac.at/RNA/](https://www.tbi.univie.ac.at/RNA/)

## 📧 联系方式

- **作者**: YANG, HENG (杨恒)
- **邮箱**: hy345@exeter.ac.uk
- **主页**: [https://yangheng95.github.io](https://yangheng95.github.io)

---

**最后更新**: 2025年11月1日
