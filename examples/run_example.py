#!/usr/bin/env python
"""GBench 测试脚本"""

import sys
from pathlib import Path

# 添加项目路径到 PYTHONPATH
example_dir = Path(__file__).parent
project_root = example_dir.parent
sys.path.insert(0, str(project_root))

from gbench import GBenchRunner
from gbench.utils import get_logger


def main():
    logger = get_logger("test")

    logger.info("=" * 80)
    logger.info("开始 GBench 测试")
    logger.info("=" * 80)

    # 加载配置
    config_file = example_dir / "example_config.yaml"

    if not config_file.exists():
        logger.error(f"配置文件不存在: {config_file}")
        logger.info("请先编辑 example_config.yaml，填入实际的模型路径和 API 配置")
        sys.exit(1)

    logger.info(f"配置文件: {config_file}")

    # 运行评测
    try:
        runner = GBenchRunner(config_file=str(config_file))
        runner.run()

        logger.info("=" * 80)
        logger.info("测试完成！")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
