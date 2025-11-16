"""推理引擎模块"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from gbench.utils import get_logger, load_jsonl, save_jsonl


class VLLMInference:
    """vLLM 推理引擎"""

    def __init__(self, logger=None):
        self.logger = logger or get_logger("vllm_inference")

    def run(
        self,
        data_file: str | Path,
        output_dir: str | Path,
        model_name: str,
        model_path: str,
        conda_env: str = "base",
        python_path: str | None = None,
        total_gpus: int = 8,
        tensor_parallel_size: int = 1,
        num_samples: int = 1,
        response_processor: Callable[[str], str] | None = None,
        extra_args: Dict[str, Any] | None = None,
    ) -> Path:
        """
        执行 vLLM 推理

        参数:
            data_file: 输入数据文件
            output_dir: 输出目录
            model_name: 模型名称（用于标识）
            model_path: 模型路径
            conda_env: conda 环境名称（如果未提供 python_path）
            python_path: Python 可执行文件路径（优先使用）
            total_gpus: 可用 GPU 总数
            tensor_parallel_size: 张量并行大小
            num_samples: 每个 prompt 采样次数
            response_processor: 响应后处理函数
            extra_args: 额外参数（如 temperature, max_tokens 等）

        返回:
            推理结果文件路径
        """
        data_file = Path(data_file)
        output_dir = Path(output_dir) / "infer" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "inference_result.jsonl"

        # 检查是否已存在结果（断点续传）
        if output_file.exists():
            existing_data = load_jsonl(output_file)
            input_data = load_jsonl(data_file)

            if len(existing_data) == len(input_data):
                # 检查所有数据是否都有 responses
                if all(item.get("responses") is not None for item in existing_data):
                    self.logger.info(f"推理结果已存在，跳过: {output_file}")
                    return output_file
                else:
                    self.logger.info(f"检测到不完整的推理结果，重新推理")

        self.logger.info(f"开始 vLLM 推理: {model_name}")
        self.logger.info(f"  - 模型路径: {model_path}")
        self.logger.info(f"  - Tensor Parallel Size: {tensor_parallel_size}")
        self.logger.info(f"  - 采样次数: {num_samples}")

        # 计算数据并行度
        data_parallel = total_gpus // tensor_parallel_size
        self.logger.info(f"  - 数据并行度: {data_parallel}")

        # 准备推理参数
        extra_args = extra_args or {}
        temperature = extra_args.get("temperature", 0.7)
        max_tokens = extra_args.get("max_tokens", 2048)
        top_p = extra_args.get("top_p", 0.95)

        if data_parallel == 1:
            # 单进程推理
            self._run_single_inference(
                data_file=data_file,
                output_file=output_file,
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                num_samples=num_samples,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                gpu_ids=list(range(tensor_parallel_size)),
                python_path=python_path,
                conda_env=conda_env,
            )
        else:
            # 数据并行推理
            self._run_parallel_inference(
                data_file=data_file,
                output_file=output_file,
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                data_parallel=data_parallel,
                num_samples=num_samples,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                python_path=python_path,
                conda_env=conda_env,
            )

        # 应用响应后处理
        if response_processor is not None:
            self.logger.info("应用响应后处理...")
            self._apply_response_processor(output_file, response_processor)

        return output_file

    def _run_single_inference(
        self,
        data_file: Path,
        output_file: Path,
        model_path: str,
        tensor_parallel_size: int,
        num_samples: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        gpu_ids: List[int],
        python_path: str | None,
        conda_env: str,
    ) -> None:
        """单进程推理"""
        # 创建推理脚本
        script_content = self._generate_vllm_script(
            data_file=str(data_file),
            output_file=str(output_file),
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            num_samples=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        # 写入临时脚本文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # 设置环境变量指定 GPU
            env_vars = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu_ids))}"

            # 执行推理脚本
            if python_path:
                cmd = f"{env_vars} {python_path} {script_path}"
            else:
                cmd = f"{env_vars} conda run -n {conda_env} python {script_path}"

            self.logger.info(f"执行命令: {cmd}")

            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )

            self.logger.info("vLLM 推理完成")
            if result.stdout:
                self.logger.debug(f"STDOUT: {result.stdout}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"vLLM 推理失败: {e}")
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR: {e.stderr}")
            raise

        finally:
            # 清理临时脚本
            Path(script_path).unlink(missing_ok=True)

    def _run_parallel_inference(
        self,
        data_file: Path,
        output_file: Path,
        model_path: str,
        tensor_parallel_size: int,
        data_parallel: int,
        num_samples: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        python_path: str | None,
        conda_env: str,
    ) -> None:
        """数据并行推理"""
        self.logger.info(f"启动 {data_parallel} 个并行进程...")

        # 加载数据
        all_data = load_jsonl(data_file)
        total_samples = len(all_data)

        # 分割数据
        chunk_size = (total_samples + data_parallel - 1) // data_parallel
        data_chunks = []
        for i in range(data_parallel):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)
            if start_idx < end_idx:
                data_chunks.append(all_data[start_idx:end_idx])

        # 创建临时目录
        temp_dir = output_file.parent / "temp"
        temp_dir.mkdir(exist_ok=True)

        # 为每个进程准备数据文件和输出文件
        tasks = []
        for rank, chunk in enumerate(data_chunks):
            chunk_input = temp_dir / f"input_rank_{rank}.jsonl"
            chunk_output = temp_dir / f"output_rank_{rank}.jsonl"

            # 保存分块数据
            save_jsonl(chunk, chunk_input)

            # 计算 GPU IDs
            gpu_start = rank * tensor_parallel_size
            gpu_ids = list(range(gpu_start, gpu_start + tensor_parallel_size))

            tasks.append(
                {
                    "rank": rank,
                    "input_file": chunk_input,
                    "output_file": chunk_output,
                    "gpu_ids": gpu_ids,
                }
            )

        # 准备所有进程的脚本
        processes = []
        for task in tasks:
            self.logger.info(
                f"准备进程 {task['rank']}: GPU {task['gpu_ids']}, "
                f"样本数 {len(load_jsonl(task['input_file']))}"
            )

            # 生成脚本
            script_content = self._generate_vllm_script(
                data_file=str(task["input_file"]),
                output_file=str(task["output_file"]),
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                num_samples=num_samples,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            # 写入脚本
            script_file = temp_dir / f"script_rank_{task['rank']}.py"
            with open(script_file, "w") as f:
                f.write(script_content)

            # 设置环境变量
            env_vars = f"CUDA_VISIBLE_DEVICES={','.join(map(str, task['gpu_ids']))}"

            # 构建命令
            if python_path:
                cmd = f"{env_vars} {python_path} {script_file}"
            else:
                cmd = f"{env_vars} conda run -n {conda_env} python {script_file}"

            processes.append(
                {
                    "rank": task["rank"],
                    "cmd": cmd,
                    "output_file": task["output_file"],
                    "process": None,
                }
            )

        # 并发启动所有进程
        self.logger.info(f"并发启动 {len(processes)} 个推理进程...")
        for proc_info in processes:
            self.logger.info(f"启动进程 {proc_info['rank']}: {proc_info['cmd']}")
            proc = subprocess.Popen(
                proc_info["cmd"],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc_info["process"] = proc
            time.sleep(1)  # 稍微延迟，避免同时初始化模型

        # 等待所有进程完成
        self.logger.info("等待所有进程完成...")
        results = []
        for proc_info in processes:
            proc = proc_info["process"]
            stdout, stderr = proc.communicate()

            if proc.returncode != 0:
                self.logger.error(f"进程 {proc_info['rank']} 失败，返回码: {proc.returncode}")
                if stderr:
                    self.logger.error(f"STDERR: {stderr}")
                raise RuntimeError(f"进程 {proc_info['rank']} 执行失败")

            self.logger.info(f"进程 {proc_info['rank']} 完成")
            results.append(load_jsonl(proc_info["output_file"]))

        # 合并结果
        self.logger.info("合并推理结果...")
        merged_data = []
        for chunk_result in results:
            merged_data.extend(chunk_result)

        # 保存合并后的结果
        save_jsonl(merged_data, output_file)

        # 清理临时文件
        import shutil

        shutil.rmtree(temp_dir)

        self.logger.info(f"数据并行推理完成，共 {len(merged_data)} 条结果")

    def _generate_vllm_script(
        self,
        data_file: str,
        output_file: str,
        model_path: str,
        tensor_parallel_size: int,
        num_samples: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> str:
        """生成 vLLM 推理脚本"""
        return f'''
import json
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 加载数据
data = []
with open("{data_file}", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))

# 初始化 vLLM
llm = LLM(
    model="{model_path}",
    tensor_parallel_size={tensor_parallel_size},
    trust_remote_code=True,
)

# 加载 tokenizer 以应用 chat template
tokenizer = AutoTokenizer.from_pretrained("{model_path}", trust_remote_code=True)

# 应用 chat template 并提取 prompts
prompts = []
for item in data:
    # 将用户输入转换为对话格式
    messages = [
        {{"role": "user", "content": item["prompt"]}}
    ]
    # 应用 chat template
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)
    except Exception as e:
        # 如果模型不支持 chat template，直接使用原始 prompt
        print(f"警告: 无法应用 chat template ({{e}})，使用原始 prompt")
        prompts.append(item["prompt"])

# 设置采样参数
sampling_params = SamplingParams(
    temperature={temperature},
    top_p={top_p},
    max_tokens={max_tokens},
    n={num_samples},
)

# 执行推理
print(f"开始推理 {{len(prompts)}} 个样本...")
outputs = llm.generate(prompts, sampling_params)

# 处理结果
for i, output in enumerate(outputs):
    responses = [o.text for o in output.outputs]
    data[i]["responses"] = responses

# 保存结果
Path("{output_file}").parent.mkdir(parents=True, exist_ok=True)
with open("{output_file}", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\\n")

print(f"推理完成，结果保存到: {output_file}")
'''

    def _apply_response_processor(self, output_file: Path, processor: Callable[[str], str]) -> None:
        """应用响应后处理函数"""
        data = load_jsonl(output_file)

        for item in data:
            if item.get("responses"):
                # 处理每个响应
                item["responses"] = [processor(resp) for resp in item["responses"]]

        save_jsonl(data, output_file)
        self.logger.info("响应后处理完成")
