#!/bin/bash

# GBench 代码质量检查和格式化脚本

cd /mnt/shared-storage-user/songdemin/user/guoxu/workspace/gbench_v2

echo "=========================================="
echo "GBench 代码检查和格式化"
echo "=========================================="
echo ""

# 检查是否安装 ruff
if ! command -v ruff &> /dev/null; then
    echo "ruff 未安装，正在安装..."
    pip install ruff
fi

echo "1. 运行 ruff check（检查代码问题）..."
echo "=========================================="
ruff check gbench/ --fix

echo ""
echo "2. 运行 ruff format（格式化代码）..."
echo "=========================================="
ruff format gbench/

echo ""
echo "3. 最终检查..."
echo "=========================================="
ruff check gbench/

echo ""
echo "=========================================="
echo "代码检查和格式化完成！"
echo "=========================================="

