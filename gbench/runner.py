"""主运行器"""

import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml
from tqdm import tqdm

from gbench.core import DataProcessor, LLMJudge, SummaryGenerator, VLLMInference
from gbench.utils import ensure_dir, get_logger


class GBenchRunner:
    """GBench 评测框架主运行器"""

    def __init__(
        self,
        config_file: str | Path | None = None,
        config_dict: Dict[str, Any] | None = None,
    ):
        """
        初始化运行器

        参数:
            config_file: 配置文件路径（YAML 格式）
            config_dict: 配置字典（直接传入配置）
        """
        if config_file:
            with open(config_file, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("必须提供 config_file 或 config_dict")

        # 设置输出目录
        self._setup_output_dir()

        # 初始化日志
        log_file = self.output_dir / "run.log"
        self.logger = get_logger("gbench", log_file)

        self.logger.info("=" * 80)
        self.logger.info(f"GBench 评测框架启动")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info("=" * 80)

    def _setup_output_dir(self) -> None:
        """设置输出目录"""
        if "output_dir" in self.config and self.config["output_dir"]:
            # 使用指定的输出目录（用于断点续传）
            self.output_dir = Path(self.config["output_dir"])
        else:
            # 创建新的输出目录（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output = Path(self.config.get("base_output_dir", "output"))
            self.output_dir = base_output / timestamp

        ensure_dir(self.output_dir)

    def run(self) -> None:
        """运行完整的评测流程"""
        # 1. 数据处理
        if self.config.get("run_data_processing", True):
            self._run_data_processing()

        # 2. 模型推理
        if self.config.get("run_inference", True):
            self._run_inference()

        # 3. 结果评测
        if self.config.get("run_judge", True):
            self._run_judge()

        # 4. 指标汇总
        if self.config.get("run_summary", True):
            self._run_summary()

        self.logger.info("=" * 80)
        self.logger.info("评测流程全部完成！")
        self.logger.info(f"结果保存在: {self.output_dir}")
        self.logger.info("=" * 80)

    def _run_data_processing(self) -> None:
        """运行数据处理"""
        self.logger.info("=" * 80)
        self.logger.info("步骤 1: 数据处理")
        self.logger.info("=" * 80)

        data_config = self.config.get("data", {})

        # 检查是否已处理
        data_file = self.output_dir / "data" / "data.jsonl"
        if data_file.exists():
            self.logger.info(f"数据已处理，跳过: {data_file}")
            return

        # 加载自定义 parser
        parser = None
        if "parser" in data_config and data_config["parser"]:
            parser_path = data_config["parser"]
            if isinstance(parser_path, str):
                parser = self._load_function_from_path(parser_path, "parse")
            else:
                parser = parser_path

        # 执行数据处理
        processor = DataProcessor(logger=self.logger)
        processor.process(
            input_file=data_config["input_file"],
            output_dir=self.output_dir,
            parser=parser,
        )

    def _run_inference(self) -> None:
        """运行模型推理"""
        self.logger.info("=" * 80)
        self.logger.info("步骤 2: 模型推理")
        self.logger.info("=" * 80)

        inference_config = self.config.get("inference", {})
        models = inference_config.get("models", [])

        if not models:
            self.logger.warning("未配置模型，跳过推理")
            return

        data_file = self.output_dir / "data" / "data.jsonl"

        # 加载响应后处理函数
        response_processor = None
        if "response_processor" in inference_config and inference_config["response_processor"]:
            processor_path = inference_config["response_processor"]
            if isinstance(processor_path, str):
                response_processor = self._load_function_from_path(processor_path, "process")
            else:
                response_processor = processor_path

        # 对每个模型执行推理
        for model_config in models:
            model_name = model_config["name"]

            self.logger.info(f"\n推理模型: {model_name}")

            # 检查推理类型
            infer_type = model_config.get("type", "vllm")
            if infer_type != "vllm":
                self.logger.error(f"不支持的推理类型: {infer_type}")
                continue

            # 执行 vLLM 推理
            inference = VLLMInference(logger=self.logger)
            inference.run(
                data_file=data_file,
                output_dir=self.output_dir,
                model_name=model_name,
                model_path=model_config["model_path"],
                conda_env=model_config.get("conda_env", "base"),
                python_path=model_config.get("python_path", None),
                total_gpus=inference_config.get("total_gpus", 8),
                tensor_parallel_size=model_config.get("tensor_parallel_size", 1),
                num_samples=model_config.get("num_samples", 1),
                response_processor=response_processor,
                extra_args=model_config.get("extra_args", {}),
            )

    def _run_judge(self) -> None:
        """运行结果评测"""
        self.logger.info("=" * 80)
        self.logger.info("步骤 3: 结果评测")
        self.logger.info("=" * 80)

        judge_config = self.config.get("judge", {})
        inference_config = self.config.get("inference", {})
        models = inference_config.get("models", [])

        if not models:
            self.logger.warning("未配置模型，跳过评测")
            return

        # 检查评测类型
        judge_type = judge_config.get("type", "llm")

        # 加载响应后处理函数（评测前处理）
        response_processor = None
        if "response_processor" in judge_config and judge_config["response_processor"]:
            processor_path = judge_config["response_processor"]
            if isinstance(processor_path, str):
                response_processor = self._load_function_from_path(processor_path, "process")
            else:
                response_processor = processor_path

        # 根据类型创建 judge
        if judge_type == "llm":
            judge = LLMJudge(logger=self.logger)
        elif judge_type == "custom":
            # 加载自定义 judge
            judge_class_path = judge_config.get("judge_class", "")
            if not judge_class_path:
                self.logger.error("自定义 judge 需要指定 judge_class")
                return
            judge = self._load_custom_judge(judge_class_path, judge_config.get("extra_args", {}))
        else:
            self.logger.error(f"不支持的评测类型: {judge_type}")
            return

        # 对每个模型执行评测
        for model_config in models:
            model_name = model_config["name"]

            self.logger.info(f"\n评测模型: {model_name}")

            inference_file = self.output_dir / "infer" / model_name / "inference_result.jsonl"

            if not inference_file.exists():
                self.logger.warning(f"推理结果不存在，跳过评测: {inference_file}")
                continue

            if judge_type == "llm":
                judge.run(
                    inference_file=inference_file,
                    output_dir=self.output_dir,
                    model_name=model_name,
                    max_workers=judge_config.get("max_workers", 10),
                    response_processor=response_processor,
                    extra_args=judge_config.get("extra_args", {}),
                )
            elif judge_type == "custom":
                # 使用自定义 judge 的运行方法
                self._run_custom_judge(
                    judge=judge,
                    inference_file=inference_file,
                    model_name=model_name,
                    max_workers=judge_config.get("max_workers", 10),
                    response_processor=response_processor,
                )

    def _run_summary(self) -> None:
        """运行指标汇总"""
        self.logger.info("=" * 80)
        self.logger.info("步骤 4: 指标汇总")
        self.logger.info("=" * 80)

        inference_config = self.config.get("inference", {})
        models = inference_config.get("models", [])
        model_names = [m["name"] for m in models]

        if not model_names:
            self.logger.warning("未配置模型，跳过汇总")
            return

        # 生成汇总报告
        summary = SummaryGenerator(logger=self.logger)
        eval_dir = self.output_dir / "eval"

        summary.run(
            eval_dir=eval_dir,
            output_dir=self.output_dir,
            model_names=model_names,
        )

    def _load_function_from_path(self, file_path: str, func_name: str) -> Callable:
        """从文件路径加载函数"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 动态导入模块
        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 查找函数
        if hasattr(module, func_name):
            return getattr(module, func_name)
        else:
            raise AttributeError(f"模块中未找到函数 '{func_name}': {file_path}")

    def _load_custom_judge(self, class_path: str, extra_args: Dict[str, Any]) -> Any:
        """加载自定义 judge 类"""
        # 解析路径: file_path:ClassName
        if ":" not in class_path:
            raise ValueError(f"judge_class 格式错误，应为 'file_path:ClassName': {class_path}")

        file_path_str, class_name = class_path.rsplit(":", 1)
        file_path = Path(file_path_str)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 动态导入模块
        spec = importlib.util.spec_from_file_location("custom_judge_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 查找类
        if not hasattr(module, class_name):
            raise AttributeError(f"模块中未找到类 '{class_name}': {file_path}")

        judge_class = getattr(module, class_name)

        # 实例化 judge
        return judge_class(**extra_args)

    def _run_custom_judge(
        self,
        judge: Any,
        inference_file: Path,
        model_name: str,
        max_workers: int,
        response_processor: Callable[[str], str] | None,
    ) -> None:
        """运行自定义 judge"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from gbench.utils import load_jsonl, save_jsonl

        output_dir = self.output_dir / "eval" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "judge_result.jsonl"
        verbose_file = output_dir / "judge_verbose.jsonl"

        # 检查是否已存在结果（断点续传）
        if output_file.exists():
            existing_data = load_jsonl(output_file)
            input_data = load_jsonl(inference_file)

            if len(existing_data) == len(input_data):
                if all(item.get("judges") is not None for item in existing_data):
                    self.logger.info(f"评测结果已存在，跳过: {output_file}")
                    return
                else:
                    self.logger.info(f"检测到不完整的评测结果，重新评测")

        self.logger.info(f"开始自定义评测: {model_name}")

        # 加载数据
        data = load_jsonl(inference_file)
        self.logger.info(f"加载了 {len(data)} 条推理结果")

        # 并发评测
        verbose_logs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for item in data:
                # 应用响应后处理（如果需要）
                responses = item.get("responses", [])
                if response_processor and responses:
                    responses = [response_processor(resp) for resp in responses]

                future = executor.submit(
                    self._judge_single_item_custom,
                    judge=judge,
                    item=item,
                    responses=responses,
                )
                futures[future] = item["id"]

            # 收集结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="评测进度"):
                item_id = futures[future]
                try:
                    judges, details = future.result()

                    # 更新数据
                    for item in data:
                        if item["id"] == item_id:
                            item["judges"] = judges
                            break

                    # 保存详细日志
                    if details:
                        verbose_logs.append(
                            {
                                "id": item_id,
                                "details": details,
                            }
                        )

                except Exception as e:
                    self.logger.error(f"评测失败 (ID: {item_id}): {e}")
                    # 失败时标记为 None
                    for item in data:
                        if item["id"] == item_id:
                            item["judges"] = None
                            break

        # 保存结果
        save_jsonl(data, output_file)
        self.logger.info(f"评测完成，结果保存到: {output_file}")

        # 保存详细日志
        if verbose_logs:
            save_jsonl(verbose_logs, verbose_file)
            self.logger.info(f"详细日志保存到: {verbose_file}")

    def _judge_single_item_custom(
        self,
        judge: Any,
        item: Dict[str, Any],
        responses: List[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """使用自定义 judge 评测单个样本"""
        prompt = item.get("prompt", "")
        ground_truth = item.get("ground_truth", "")

        judges = []
        details_list = []

        for response in responses:
            # 调用自定义 judge 的 judge 方法
            score, details = judge.judge(prompt, response, ground_truth)

            judges.append(
                {
                    "score": score,
                    "details": details,
                }
            )

            details_list.append(
                {
                    "response": response,
                    "score": score,
                    "details": details,
                }
            )

        return judges, details_list


def run_from_config(config_file: str | Path) -> None:
    """从配置文件运行评测"""
    runner = GBenchRunner(config_file=config_file)
    runner.run()


def run_from_dict(config_dict: Dict[str, Any]) -> None:
    """从配置字典运行评测"""
    runner = GBenchRunner(config_dict=config_dict)
    runner.run()
