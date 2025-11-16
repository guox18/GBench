"""数据处理模块"""

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List

from gbench.utils import get_logger, load_jsonl, save_jsonl


class DataProcessor:
    """数据处理器：将原始数据转换为标准评测格式"""

    def __init__(self, logger=None):
        self.logger = logger or get_logger("data_processor")

    def process(
        self,
        input_file: str | Path,
        output_dir: str | Path,
        parser: str | Callable | None = None,
    ) -> Path:
        """
        处理输入数据，转换为标准格式

        参数:
            input_file: 输入数据文件路径
            output_dir: 输出目录
            parser: 自定义解析器（函数或模块路径）

        返回:
            处理后的数据文件路径
        """
        self.logger.info(f"开始处理数据: {input_file}")

        # 加载原始数据
        raw_data = load_jsonl(input_file)
        self.logger.info(f"加载了 {len(raw_data)} 条原始数据")

        # 应用解析器
        if parser is None:
            # 默认解析器：假设数据已经是标准格式或接近标准格式
            processed_data = self._default_parser(raw_data)
        elif callable(parser):
            # 直接使用传入的函数
            processed_data = [parser(item) for item in raw_data]
        elif isinstance(parser, str):
            # 从路径加载解析器函数
            parser_func = self._load_parser_from_path(parser)
            processed_data = [parser_func(item) for item in raw_data]
        else:
            raise ValueError(f"不支持的 parser 类型: {type(parser)}")

        # 验证数据格式
        self._validate_data(processed_data)

        # 保存处理后的数据
        output_file = Path(output_dir) / "data" / "data.jsonl"
        save_jsonl(processed_data, output_file)

        self.logger.info(f"数据处理完成，保存到: {output_file}")
        return output_file

    def _default_parser(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        默认解析器：标准化数据格式

        标准格式:
        {
            "id": int,
            "prompt": str,
            "ground_truth": str (可选),
            "metadata": dict (可选),
            "responses": null,
            "judges": null
        }
        """
        processed = []
        for i, item in enumerate(raw_data):
            processed.append(
                {
                    "id": item.get("id", i),
                    "prompt": item.get("prompt", item.get("question", "")),
                    "ground_truth": item.get("ground_truth", item.get("answer", "")),
                    "metadata": item.get("metadata", {}),
                    "responses": None,
                    "judges": None,
                }
            )
        return processed

    def _load_parser_from_path(self, parser_path: str) -> Callable:
        """从文件路径加载解析器函数"""
        parser_path = Path(parser_path)

        if not parser_path.exists():
            raise FileNotFoundError(f"解析器文件不存在: {parser_path}")

        # 动态导入模块
        spec = importlib.util.spec_from_file_location("custom_parser", parser_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载解析器模块: {parser_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 查找 parse 函数
        if hasattr(module, "parse"):
            return module.parse
        else:
            raise AttributeError(f"解析器模块中未找到 'parse' 函数: {parser_path}")

    def _validate_data(self, data: List[Dict[str, Any]]) -> None:
        """验证数据格式是否符合标准"""
        required_fields = ["id", "prompt"]

        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"第 {i} 条数据缺少必填字段: {field}")

            # 确保字段存在（即使为 None）
            if "responses" not in item:
                item["responses"] = None
            if "judges" not in item:
                item["judges"] = None
