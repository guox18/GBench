"""命令行接口"""

import argparse
import sys
from pathlib import Path

from gbench import GBenchRunner
from gbench.utils import get_logger


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="GBench - 通用大模型评测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从配置文件运行完整评测
  gbench run --config config.yaml
  
  # 断点续传
  gbench run --config config.yaml --output-dir output/20231201_120000
  
  # 只运行特定步骤
  gbench run --config config.yaml --skip-inference --skip-judge
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run 命令
    run_parser = subparsers.add_parser("run", help="运行评测")
    run_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="配置文件路径",
    )
    run_parser.add_argument(
        "--output-dir",
        "-o",
        help="输出目录（用于断点续传）",
    )
    run_parser.add_argument(
        "--skip-data",
        action="store_true",
        help="跳过数据处理",
    )
    run_parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="跳过模型推理",
    )
    run_parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="跳过结果评测",
    )
    run_parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="跳过指标汇总",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        run_evaluation(args)


def run_evaluation(args):
    """运行评测"""
    logger = get_logger("gbench.cli")

    config_file = Path(args.config)
    if not config_file.exists():
        logger.error(f"配置文件不存在: {config_file}")
        sys.exit(1)

    try:
        # 加载配置
        import yaml

        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 覆盖配置
        if args.output_dir:
            config["output_dir"] = args.output_dir

        if args.skip_data:
            config["run_data_processing"] = False
        if args.skip_inference:
            config["run_inference"] = False
        if args.skip_judge:
            config["run_judge"] = False
        if args.skip_summary:
            config["run_summary"] = False

        # 运行评测
        runner = GBenchRunner(config_dict=config)
        runner.run()

        logger.info("评测完成！")

    except Exception as e:
        logger.error(f"评测失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
