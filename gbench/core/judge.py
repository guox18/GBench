"""评测模块"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List

from jinja2 import Template
from tqdm import tqdm

from gbench.utils import get_logger, load_jsonl, save_jsonl


class LLMJudge:
    """LLM 评测器"""

    def __init__(self, logger=None):
        self.logger = logger or get_logger("llm_judge")

    def run(
        self,
        inference_file: str | Path,
        output_dir: str | Path,
        model_name: str,
        max_workers: int = 10,
        response_processor: Callable[[str], str] | None = None,
        extra_args: Dict[str, Any] | None = None,
    ) -> Path:
        """
        执行 LLM 评测

        参数:
            inference_file: 推理结果文件
            output_dir: 输出目录
            model_name: 模型名称
            max_workers: 并发线程数
            response_processor: 响应后处理函数（对评测前的 response 进行处理）
            extra_args: 额外参数（API 配置等）

        返回:
            评测结果文件路径
        """
        inference_file = Path(inference_file)
        output_dir = Path(output_dir) / "eval" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "judge_result.jsonl"
        verbose_file = output_dir / "judge_verbose.jsonl"

        # 检查是否已存在结果（断点续传）
        if output_file.exists():
            existing_data = load_jsonl(output_file)
            input_data = load_jsonl(inference_file)

            if len(existing_data) == len(input_data):
                # 检查所有数据是否都有 judges
                if all(item.get("judges") is not None for item in existing_data):
                    self.logger.info(f"评测结果已存在，跳过: {output_file}")
                    return output_file
                else:
                    self.logger.info(f"检测到不完整的评测结果，重新评测")

        self.logger.info(f"开始 LLM 评测: {model_name}")

        # 加载数据
        data = load_jsonl(inference_file)
        self.logger.info(f"加载了 {len(data)} 条推理结果")

        # 加载评测模板
        template = self._load_template(extra_args)

        # 配置 API 客户端
        api_config = self._parse_api_config(extra_args)

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
                    self._judge_single_item,
                    item=item,
                    responses=responses,
                    template=template,
                    api_config=api_config,
                )
                futures[future] = item["id"]

            # 收集结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="评测进度"):
                item_id = futures[future]
                try:
                    judges, verbose_log = future.result()

                    # 更新数据
                    for item in data:
                        if item["id"] == item_id:
                            item["judges"] = judges
                            break

                    # 保存详细日志
                    if verbose_log:
                        verbose_logs.append(
                            {
                                "id": item_id,
                                "log": verbose_log,
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

        return output_file

    def _load_template(self, extra_args: Dict[str, Any] | None) -> Template:
        """加载评测模板"""
        if extra_args and "template_file" in extra_args:
            template_file = Path(extra_args["template_file"])
            with open(template_file, "r", encoding="utf-8") as f:
                template_content = f.read()
        else:
            # 使用内置模板
            template_file = Path(__file__).parent.parent / "templates" / "judge_template.j2"
            with open(template_file, "r", encoding="utf-8") as f:
                template_content = f.read()

        return Template(template_content)

    def _parse_api_config(self, extra_args: Dict[str, Any] | None) -> Dict[str, Any]:
        """解析 API 配置"""
        if not extra_args:
            return {}

        return {
            "api_base": extra_args.get("api_base", "https://api.openai.com/v1"),
            "api_key": extra_args.get("api_key", ""),
            "model": extra_args.get("judge_model", "gpt-4"),
            "temperature": extra_args.get("temperature", 0.0),
            "max_tokens": extra_args.get("max_tokens", 10),
        }

    def _judge_single_item(
        self,
        item: Dict[str, Any],
        responses: List[str],
        template: Template,
        api_config: Dict[str, Any],
    ) -> tuple[List[int], List[Dict[str, Any]]]:
        """评测单个样本"""
        prompt = item.get("prompt", "")
        ground_truth = item.get("ground_truth", "")

        judges = []
        verbose_log = []

        for response in responses:
            # 构建评测 prompt
            judge_prompt = template.render(
                origin_question=prompt,
                gold_target=ground_truth,
                predicted_answer=response,
            )

            # 调用 LLM API
            judge_response = self._call_llm_api(judge_prompt, api_config)

            # 解析评测结果
            score = self._parse_judge_result(judge_response)
            judges.append(score)

            # 记录详细日志
            verbose_log.append(
                {
                    "response": response,
                    "judge_prompt": judge_prompt,
                    "judge_response": judge_response,
                    "score": score,
                }
            )

        return judges, verbose_log

    def _call_llm_api(
        self,
        prompt: str,
        api_config: Dict[str, Any],
    ) -> str:
        """调用 LLM API"""
        import openai

        client = openai.OpenAI(
            api_key=api_config.get("api_key"),
            base_url=api_config.get("api_base"),
        )

        response = client.chat.completions.create(
            model=api_config.get("model", "gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            temperature=api_config.get("temperature", 0.0),
            max_tokens=api_config.get("max_tokens", 10),
        )

        return response.choices[0].message.content.strip()

    def _parse_judge_result(self, response: str) -> int:
        """
        解析评测结果，返回 1（正确）或 0（错误）

        支持的格式:
        - "CORRECT" / "A" -> 1
        - "INCORRECT" / "B" -> 0
        """
        response = response.strip().upper()

        # 检查是否包含 CORRECT
        if "CORRECT" in response and "INCORRECT" not in response:
            return 1

        # 检查是否包含 INCORRECT
        if "INCORRECT" in response:
            return 0

        # 检查是否只包含字母 A 或 B
        if re.match(r"^A$", response):
            return 1
        if re.match(r"^B$", response):
            return 0

        # 默认返回 0（保守策略）
        self.logger.warning(f"无法解析评测结果: {response}，默认返回 0")
        return 0
