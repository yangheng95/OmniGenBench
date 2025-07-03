# OmniGenomeBench 测试套件

本文档描述了 OmniGenomeBench 项目的测试套件，包括如何运行测试、测试覆盖范围和使用指南。

## 概述

OmniGenomeBench 测试套件提供了全面的单元测试和集成测试，确保代码质量和功能正确性。测试套件包括：

- **单元测试**: 测试各个组件的独立功能
- **集成测试**: 测试组件间的协作
- **错误处理测试**: 测试异常情况的处理
- **性能测试**: 测试关键功能的性能

## 测试结构

```
tests/
├── __init__.py                 # 测试包初始化
├── conftest.py                 # pytest 配置和通用 fixtures
├── test_abstract_model.py      # 抽象模型类测试
├── test_dataset.py             # 数据集类测试
├── test_tokenizer.py           # 分词器类测试
├── test_metric.py              # 评估指标类测试
├── test_utils.py               # 工具函数测试
├── test_integration.py         # 集成测试
└── test_auto_bench.py          # AutoBench 测试
```

## 安装测试依赖

在运行测试之前，请确保安装了所有必要的依赖：

```bash
# 安装测试依赖
pip install pytest pytest-cov pytest-mock

# 或者从 requirements.txt 安装
pip install -r requirements.txt
```

## 运行测试

### 运行所有测试

```bash
# 运行所有测试
pytest

# 运行测试并显示详细输出
pytest -v

# 运行测试并显示覆盖率报告
pytest --cov=omnigenome --cov-report=html
```

### 运行特定测试

```bash
# 运行特定测试文件
pytest tests/test_dataset.py

# 运行特定测试类
pytest tests/test_dataset.py::TestOmniDatasetForSequenceClassification

# 运行特定测试方法
pytest tests/test_dataset.py::TestOmniDatasetForSequenceClassification::test_init

# 运行标记的测试
pytest -m "unit"           # 运行单元测试
pytest -m "integration"    # 运行集成测试
pytest -m "not slow"       # 运行非慢速测试
```

### 测试覆盖率

```bash
# 生成覆盖率报告
pytest --cov=omnigenome --cov-report=html --cov-report=term-missing

# 覆盖率报告将生成在 htmlcov/ 目录中
# 打开 htmlcov/index.html 查看详细报告
```

## 测试标记

测试套件使用以下标记来分类测试：

- `@pytest.mark.unit`: 单元测试
- `@pytest.mark.integration`: 集成测试
- `@pytest.mark.slow`: 慢速测试
- `@pytest.mark.gpu`: 需要 GPU 的测试
- `@pytest.mark.cpu`: 仅 CPU 测试

## 测试 Fixtures

测试套件提供了多个通用的 fixtures：

### 数据 Fixtures

- `sample_sequences`: 示例 RNA/DNA 序列
- `sample_labels`: 示例标签
- `sample_token_labels`: 示例 token 级标签
- `sample_dataset_json`: 示例数据集 JSON 文件

### Mock Fixtures

- `mock_tokenizer`: 模拟分词器
- `mock_config`: 模拟模型配置
- `mock_model`: 模拟基础模型

### 环境 Fixtures

- `test_data_dir`: 临时测试数据目录
- `device`: 测试设备 (CPU/GPU)
- `set_seed`: 设置随机种子

## 测试用例说明

### 1. 抽象模型测试 (`test_abstract_model.py`)

测试 `OmniModel` 基类的功能：

- 模型初始化（字符串、对象、配置）
- 前向传播
- 损失函数计算
- 模型保存和加载
- 元数据管理

```python
def test_init_with_string_model(self, mock_tokenizer, mock_config):
    """测试使用字符串路径初始化模型"""
    model = OmniModel("test_model", mock_tokenizer, num_labels=2)
    assert model.config == mock_config
    assert model.tokenizer == mock_tokenizer
```

### 2. 数据集测试 (`test_dataset.py`)

测试各种数据集类的功能：

- 序列分类数据集
- Token 分类数据集
- 序列回归数据集
- Token 回归数据集

```python
def test_prepare_input_with_dict(self, mock_tokenizer):
    """测试使用字典输入准备数据"""
    dataset = OmniDatasetForSequenceClassification(
        data_source="test_data",
        tokenizer=mock_tokenizer,
        max_length=512
    )
    instance = {"seq": "AUGGCCUAA", "label": 1}
    result = dataset.prepare_input(instance)
    assert result["labels"] == torch.tensor(1)
```

### 3. 分词器测试 (`test_tokenizer.py`)

测试 `OmniSingleNucleotideTokenizer` 的功能：

- 基本分词功能
- U/T 转换
- 空格添加
- 最大长度截断

```python
def test_call_with_u2t_conversion(self):
    """测试 U 到 T 的转换"""
    tokenizer = OmniSingleNucleotideTokenizer(base_tokenizer, u2t=True)
    result = tokenizer("AUGGCCUAA")
    assert "input_ids" in result
    assert "attention_mask" in result
```

### 4. 评估指标测试 (`test_metric.py`)

测试评估指标类的功能：

- MCRMSE 计算
- 分类指标（准确率、精确率、召回率等）
- 回归指标（MAE、MSE、RMSE 等）
- 忽略值处理

```python
def test_mcrmse_basic(self):
    """测试基本 MCRMSE 计算"""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 2.1], [3.1, 4.1]])
    result = mcrmse(y_true, y_pred)
    assert isinstance(result, float)
    assert result > 0
```

### 5. 工具函数测试 (`test_utils.py`)

测试工具函数的功能：

- 格式化打印
- 环境元信息收集
- 临时文件清理

```python
def test_fprint_basic(self, capsys):
    """测试基本格式化打印"""
    test_message = "Test message"
    fprint(test_message)
    captured = capsys.readouterr()
    assert test_message in captured.out
```

### 6. 集成测试 (`test_integration.py`)

测试组件间的协作：

- 模型-数据集-分词器集成
- 端到端分类流程
- 错误处理集成
- 性能测试

```python
def test_end_to_end_classification_pipeline(self, mock_tokenizer, mock_model):
    """测试端到端分类流程"""
    model = OmniModel(mock_model, mock_tokenizer, num_labels=2)
    dataset = OmniDatasetForSequenceClassification(...)
    metric = Metric()
    
    # 测试完整流程
    for instance in test_instances:
        prepared_input = dataset.prepare_input(instance)
        result = model(**prepared_input)
        # 评估结果
```

### 7. AutoBench 测试 (`test_auto_bench.py`)

测试自动基准测试功能：

- 配置管理
- 基准测试运行
- 结果管理
- 错误处理

```python
def test_run_basic(self):
    """测试基本运行功能"""
    auto_bench = AutoBench(
        benchmark="RGB",
        model_name_or_path="test_model",
        device="cpu"
    )
    auto_bench.run()
    # 验证结果
```

## 编写新测试

### 添加新的测试文件

1. 在 `tests/` 目录下创建新的测试文件
2. 文件名格式：`test_<module_name>.py`
3. 导入必要的模块和 fixtures

```python
import pytest
from unittest.mock import Mock, patch
from omnigenome.src.module import ModuleClass

class TestModuleClass:
    def test_some_functionality(self, mock_fixture):
        # 测试代码
        pass
```

### 添加新的 Fixtures

在 `conftest.py` 中添加新的 fixtures：

```python
@pytest.fixture(scope="session")
def new_fixture():
    """新 fixture 的描述"""
    # fixture 实现
    return fixture_value
```

### 测试最佳实践

1. **测试命名**: 使用描述性的测试名称
2. **测试隔离**: 每个测试应该独立运行
3. **Mock 使用**: 适当使用 mock 来隔离依赖
4. **错误测试**: 测试异常情况
5. **边界测试**: 测试边界条件

```python
def test_function_with_valid_input(self):
    """测试有效输入"""
    result = function(valid_input)
    assert result == expected_output

def test_function_with_invalid_input(self):
    """测试无效输入"""
    with pytest.raises(ValueError):
        function(invalid_input)

def test_function_with_edge_cases(self):
    """测试边界情况"""
    result = function(edge_case_input)
    assert result is not None
```

## 持续集成

测试套件配置为与 CI/CD 系统集成：

- 自动运行测试
- 生成覆盖率报告
- 测试结果通知
- 代码质量检查

## 故障排除

### 常见问题

1. **导入错误**: 确保安装了所有依赖
2. **Mock 错误**: 检查 mock 对象的设置
3. **设备错误**: 确保测试环境支持所需的设备
4. **权限错误**: 检查文件系统权限

### 调试测试

```bash
# 运行单个测试并显示详细输出
pytest tests/test_file.py::test_function -v -s

# 使用 pdb 调试
pytest tests/test_file.py::test_function --pdb

# 显示测试执行时间
pytest --durations=10
```

## 贡献指南

在贡献代码时，请确保：

1. 为新功能添加相应的测试
2. 所有测试都能通过
3. 测试覆盖率不低于 80%
4. 遵循测试命名和结构约定

## 联系信息

如有测试相关的问题或建议，请：

1. 查看现有测试用例
2. 提交 Issue 描述问题
3. 提交 Pull Request 包含测试修复

---

*最后更新: 2024年* 