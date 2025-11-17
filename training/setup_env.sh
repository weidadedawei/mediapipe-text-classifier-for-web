#!/bin/bash
# 设置 TensorFlow BERT 训练环境（使用 Conda）

set -e

echo "=========================================="
echo "设置 TensorFlow BERT 训练环境"
echo "=========================================="
echo ""

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到 conda，请先安装 Anaconda 或 Miniconda"
    echo ""
    echo "安装方法："
    echo "  - Anaconda: https://www.anaconda.com/download"
    echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

ENV_NAME="bert_gpu_env"
PYTHON_VERSION="3.11"

echo "📦 创建 Conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "   ⚠️  环境已存在，将重新创建..."
    conda env remove -n $ENV_NAME -y
fi

conda create -n $ENV_NAME python=$PYTHON_VERSION -y

if [ $? -ne 0 ]; then
    echo "❌ 环境创建失败"
    exit 1
fi

echo ""
echo "🔧 安装依赖..."
echo ""

# 激活环境并安装依赖
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "❌ 环境激活失败"
    exit 1
fi

python -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    exit 1
fi

echo ""
echo "✅ 环境设置完成！"
echo ""
echo "使用方法："
echo "  conda activate $ENV_NAME"
echo ""
echo "验证安装："
echo "  python -c 'import tensorflow as tf; import transformers; import numpy as np; print(\"TensorFlow:\", tf.__version__); print(\"Transformers:\", transformers.__version__); print(\"NumPy:\", np.__version__)'"
echo ""
