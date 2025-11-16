#!/bin/bash

# GBench 安装脚本

set -e

echo "=========================================="
echo "GBench 安装脚本"
echo "=========================================="
echo ""

# 检查 Python 版本
echo "检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前 Python 版本: $python_version"

# 安装依赖
echo ""
echo "安装依赖..."
pip install -e .

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "快速开始:"
echo "  1. 查看示例配置: examples/config.yaml"
echo "  2. 运行示例: python examples/run_example.py"
echo "  3. 或使用命令行: gbench run --config examples/config.yaml"
echo ""
echo "文档: README.md"
echo ""

