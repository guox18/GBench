# GBench 使用指南

## 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [配置详解](#配置详解)
4. [自定义扩展](#自定义扩展)
5. [断点续传](#断点续传)
6. [常见问题](#常见问题)

---

## 安装

### 基础安装

```bash
pip install -e .
```

### 依赖说明

**必需依赖**：
- `pyyaml>=6.0`: 配置文件解析
- `jinja2>=3.1.0`: 评测模板渲染
- `openai>=1.0.0`: LLM API 调用

**推理依赖**（需单独安装）：
- `vllm`: 用于模型推理

---

## 快速开始

### 1. 准备数据

创建数据文件 `data/questions.jsonl`：

```jsonl
{"prompt": "什么是机器学习？", "ground_truth": "机器学习是..."}
{"prompt": "什么是深度学习？", "ground_truth": "深度学习是..."}
```

### 2. 创建配置文件

创建 `config.yaml`：

```yaml
base_output_dir: "output"
output_dir: null

data:
  input_file: "data/questions.jsonl"

inference:
  total_gpus: 8
  models:
    - name: "qwen-7b"
      model_path: "/path/to/qwen-7b"
      type: "vllm"
      conda_env: "vllm"
      tensor_parallel_size: 1
      num_samples: 1

judge:
  type: "llm"
  max_workers: 10
  extra_args:
    api_base: "https://api.openai.com/v1"
    api_key: "your-api-key"
    judge_model: "gpt-4"
```

### 3. 运行评测

#### 方式 1: 命令行

```bash
gbench run --config config.yaml
```

#### 方式 2: Python 代码

```python
from gbench import GBenchRunner

runner = GBenchRunner(config_file="config.yaml")
runner.run()
```

---

## 配置详解

### 数据处理配置

```yaml
data:
  # 输入数据文件
  input_file: "data/questions.jsonl"
  
  # 自定义解析器（可选）
  parser: "utils/custom_parser.py"
```

**自定义解析器**：

```python
# utils/custom_parser.py
def parse(raw_data: dict) -> dict:
    return {
        "id": raw_data["id"],
        "prompt": raw_data["question"],
        "ground_truth": raw_data["answer"],
        "metadata": {},
        "responses": None,
        "judges": None,
    }
```

### 推理配置

```yaml
inference:
  # GPU 总数
  total_gpus: 8
  
  # 响应后处理（移除思考过程）
  response_processor: "utils/cut_thinking.py"
  
  models:
    - name: "model-name"
      model_path: "/path/to/model"
      type: "vllm"
      conda_env: "vllm"
      
      # 张量并行大小（根据模型大小调整）
      tensor_parallel_size: 1
      
      # 每个 prompt 采样次数
      num_samples: 1
      
      # vLLM 参数
      extra_args:
        temperature: 0.7
        max_tokens: 2048
        top_p: 0.95
```

**并行策略**：

- 数据并行度 = `total_gpus / tensor_parallel_size`
- 例如：8 GPUs，TP=2，则 DP=4（4 个进程并行推理）

### 评测配置

```yaml
judge:
  type: "llm"
  
  # 并发线程数
  max_workers: 10
  
  # 评测前对响应进行处理
  response_processor: "utils/cut_thinking.py"
  
  extra_args:
    # API 配置
    api_base: "https://api.openai.com/v1"
    api_key: "your-api-key"
    judge_model: "gpt-4"
    
    # 采样参数
    temperature: 0.0
    max_tokens: 10
    
    # 自定义模板
    template_file: "templates/custom_judge.j2"
```

### 流程控制

```yaml
# 控制运行哪些步骤
run_data_processing: true
run_inference: true
run_judge: true
run_summary: true
```

---

## 自定义扩展

### 1. 自定义数据解析器

用于将各种格式的数据转换为标准格式。

```python
# my_parser.py
def parse(raw_data: dict) -> dict:
    """
    转换为标准格式:
    {
        "id": int,
        "prompt": str,
        "ground_truth": str,
        "metadata": dict,
        "responses": None,
        "judges": None
    }
    """
    return {
        "id": raw_data.get("id", 0),
        "prompt": raw_data["question"],
        "ground_truth": raw_data["answer"],
        "metadata": {"source": raw_data.get("source", "")},
        "responses": None,
        "judges": None,
    }
```

配置：

```yaml
data:
  parser: "my_parser.py"
```

### 2. 自定义响应处理器

用于处理模型输出，如移除思考过程、格式化等。

```python
# my_processor.py
def process(response: str) -> str:
    """处理模型响应"""
    # 移除思考过程
    if "<think>" in response and "</think>" in response:
        response = response.split("</think>")[-1].strip()
    
    # 提取答案
    if "答案:" in response:
        response = response.split("答案:")[-1].strip()
    
    return response
```

配置（可在 inference 或 judge 中使用）：

```yaml
inference:
  response_processor: "my_processor.py"

judge:
  response_processor: "my_processor.py"
```

### 3. 自定义评测模板

用于自定义 LLM 评测的 prompt。

```jinja2
{# custom_judge.j2 #}
你是一位专业的评测专家。请判断以下答案是否正确。

问题：{{origin_question}}

标准答案：{{gold_target}}

学生答案：{{predicted_answer}}

请仅回答 "CORRECT" 或 "INCORRECT"，不要有其他内容。

判断结果：
```

配置：

```yaml
judge:
  extra_args:
    template_file: "custom_judge.j2"
```

---

## 断点续传

GBench 支持在任何阶段中断后继续执行。

### 自动断点续传

框架会自动检测已完成的步骤：

- 如果 `data/data.jsonl` 存在，跳过数据处理
- 如果 `infer/{model}/inference_result.jsonl` 存在且完整，跳过推理
- 如果 `eval/{model}/judge_result.jsonl` 存在且完整，跳过评测

### 手动控制

#### 1. 指定输出目录

```yaml
output_dir: "output/20231201_120000"
```

或命令行：

```bash
gbench run --config config.yaml --output-dir output/20231201_120000
```

#### 2. 控制运行步骤

```yaml
run_data_processing: false  # 跳过
run_inference: false        # 跳过
run_judge: true             # 执行
run_summary: true           # 执行
```

或命令行：

```bash
gbench run --config config.yaml --skip-data --skip-inference
```

### 典型场景

**场景 1: 推理中断**

```bash
# 继续推理（自动检测已完成的模型）
gbench run --config config.yaml --output-dir output/20231201_120000
```

**场景 2: 添加新模型**

修改配置，添加新模型，然后：

```bash
gbench run --config config.yaml --output-dir output/20231201_120000 --skip-data
```

**场景 3: 重新评测**

删除评测结果，重新运行：

```bash
rm -rf output/20231201_120000/eval
gbench run --config config.yaml --output-dir output/20231201_120000 --skip-data --skip-inference
```

---

## 常见问题

### Q1: 如何支持多次采样？

设置 `num_samples` 大于 1：

```yaml
inference:
  models:
    - name: "qwen-7b"
      num_samples: 5  # 每个 prompt 采样 5 次
```

这样可以计算 `pass@5`、`mean@5` 等指标。

### Q2: 如何使用本地 API？

修改 API 配置：

```yaml
judge:
  extra_args:
    api_base: "http://localhost:8000/v1"
    api_key: "dummy"
    judge_model: "local-model"
```

### Q3: 如何调整 GPU 分配？

通过 `tensor_parallel_size` 控制：

```yaml
inference:
  total_gpus: 8
  models:
    - name: "small-model"
      tensor_parallel_size: 1  # 使用 1 GPU，8 个进程并行
    
    - name: "large-model"
      tensor_parallel_size: 4  # 使用 4 GPU，2 个进程并行
```

### Q4: 如何查看详细日志？

日志保存在 `output/{timestamp}/run.log`。

也可以查看评测详细日志：`eval/{model}/judge_verbose.jsonl`

### Q5: 推理失败怎么办？

1. 检查日志：`output/{timestamp}/run.log`
2. 检查 conda 环境是否正确
3. 检查模型路径是否正确
4. 检查 GPU 是否可用
5. 尝试减小 `tensor_parallel_size`

### Q6: 如何批量评测多个模型？

在配置文件中添加多个模型：

```yaml
inference:
  models:
    - name: "model-1"
      model_path: "/path/to/model1"
      tensor_parallel_size: 1
    
    - name: "model-2"
      model_path: "/path/to/model2"
      tensor_parallel_size: 2
    
    - name: "model-3"
      model_path: "/path/to/model3"
      tensor_parallel_size: 4
```

框架会按顺序推理，然后统一评测和汇总。

### Q7: 如何自定义评测指标？

目前支持的指标：`pass@k`、`all@k`、`mean@k`、`max@k`、`min@k`

如需自定义指标，可以修改 `gbench/core/summary.py` 中的 `_calculate_metrics` 方法。

---

## 更多资源

- [项目设计文档](project.md)
- [示例代码](examples/)


