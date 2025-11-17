"""指标汇总模块"""

import csv
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from gbench.utils import ensure_dir, get_logger, load_jsonl


class SummaryGenerator:
    """指标汇总生成器"""

    def __init__(self, logger=None):
        self.logger = logger or get_logger("summary")

    def run(
        self,
        eval_dir: str | Path,
        output_dir: str | Path,
        model_names: List[str],
    ) -> tuple[Path, Path]:
        """
        生成汇总报告

        参数:
            eval_dir: 评测结果目录
            output_dir: 输出目录
            model_names: 模型名称列表

        返回:
            (markdown 文件路径, csv 文件路径)
        """
        self.logger.info("开始生成汇总报告...")

        eval_dir = Path(eval_dir)
        output_dir = Path(output_dir) / "summary"
        ensure_dir(output_dir)

        # 收集所有模型的评测结果
        all_results = {}
        for model_name in tqdm(model_names, desc="加载模型评测结果", unit="模型"):
            judge_file = eval_dir / model_name / "judge_result.jsonl"

            if not judge_file.exists():
                tqdm.write(f"警告: 未找到评测结果: {judge_file}")
                continue

            data = load_jsonl(judge_file)
            all_results[model_name] = data
            tqdm.write(f"✓ 已加载模型 {model_name}: {len(data)} 条数据")

        if not all_results:
            self.logger.error("没有找到任何评测结果")
            raise FileNotFoundError("没有找到任何评测结果")

        # 计算指标
        metrics = {}
        for model_name, data in all_results.items():
            metrics[model_name] = self._calculate_metrics(data)

        # 生成 Markdown 报告
        md_file = output_dir / "summary.md"
        self._generate_markdown(metrics, md_file)
        self.logger.info(f"Markdown 报告保存到: {md_file}")

        # 生成 CSV 报告
        csv_file = output_dir / "summary.csv"
        self._generate_csv(metrics, csv_file)
        self.logger.info(f"CSV 报告保存到: {csv_file}")

        return md_file, csv_file

    def _extract_score(self, judge_item: Any) -> float:
        """
        从 judge 结果中提取分数
        支持两种格式:
        1. 整数: 0 或 1
        2. 字典: {"score": 1, "details": {...}}
        """
        if isinstance(judge_item, dict):
            return float(judge_item.get("score", 0))
        return float(judge_item)

    def _calculate_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算评测指标

        支持的指标:
        - pass@k: 前 k 次中至少一次正确的比例
        - all@k: 前 k 次全部正确的比例
        - mean@k: 前 k 次的平均分
        - max@k: 前 k 次最高分的平均
        - min@k: 前 k 次最低分的平均
        """
        # 过滤掉没有 judges 的数据
        valid_data = [item for item in data if item.get("judges") is not None]

        if not valid_data:
            self.logger.warning("没有有效的评测结果")
            return {}

        # 确定最大采样次数
        max_samples = max(len(item["judges"]) for item in valid_data)

        metrics = {}

        # 计算各种指标
        for k in range(1, max_samples + 1):
            # 提取分数（支持字典格式）
            scores_list = [
                [self._extract_score(j) for j in item["judges"][:k]] for item in valid_data
            ]

            # pass@k: 前 k 次中至少一次正确
            pass_at_k = sum(1 for scores in scores_list if any(scores)) / len(valid_data)
            metrics[f"pass@{k}"] = pass_at_k

            # all@k: 前 k 次全部正确
            all_at_k = sum(1 for scores in scores_list if all(scores)) / len(valid_data)
            metrics[f"all@{k}"] = all_at_k

            # mean@k: 前 k 次的平均分
            mean_at_k = sum(sum(scores) / k for scores in scores_list) / len(valid_data)
            metrics[f"mean@{k}"] = mean_at_k

            # max@k: 前 k 次最高分的平均
            max_at_k = sum(max(scores) for scores in scores_list) / len(valid_data)
            metrics[f"max@{k}"] = max_at_k

            # min@k: 前 k 次最低分的平均
            min_at_k = sum(min(scores) for scores in scores_list) / len(valid_data)
            metrics[f"min@{k}"] = min_at_k

        return metrics

    def _generate_markdown(
        self,
        metrics: Dict[str, Dict[str, float]],
        output_file: Path,
    ) -> None:
        """生成 Markdown 报告"""
        # 获取所有指标名称（保持顺序）
        metric_names = []
        if metrics:
            first_model = next(iter(metrics.values()))
            metric_names = list(first_model.keys())

        # 构建表格
        lines = []
        lines.append("# 评测结果汇总\n")
        lines.append("## 指标说明\n")
        lines.append("- `pass@k`: 前 k 次中至少一次正确的比例")
        lines.append("- `all@k`: 前 k 次全部正确的比例")
        lines.append("- `mean@k`: 前 k 次的平均分")
        lines.append("- `max@k`: 前 k 次最高分的平均")
        lines.append("- `min@k`: 前 k 次最低分的平均\n")
        lines.append("## 评测结果\n")

        # 表头
        header = "| 模型 | " + " | ".join(metric_names) + " |"
        separator = "|" + "|".join(["---"] * (len(metric_names) + 1)) + "|"

        lines.append(header)
        lines.append(separator)

        # 数据行
        for model_name, model_metrics in metrics.items():
            row = f"| {model_name} |"
            for metric_name in metric_names:
                value = model_metrics.get(metric_name, 0.0)
                row += f" {value:.4f} |"
            lines.append(row)

        # 写入文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _generate_csv(
        self,
        metrics: Dict[str, Dict[str, float]],
        output_file: Path,
    ) -> None:
        """生成 CSV 报告"""
        # 获取所有指标名称
        metric_names = []
        if metrics:
            first_model = next(iter(metrics.values()))
            metric_names = list(first_model.keys())

        # 写入 CSV
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            # 表头
            writer.writerow(["model"] + metric_names)

            # 数据行
            for model_name, model_metrics in metrics.items():
                row = [model_name]
                for metric_name in metric_names:
                    value = model_metrics.get(metric_name, 0.0)
                    row.append(f"{value:.4f}")
                writer.writerow(row)
