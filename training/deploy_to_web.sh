#!/bin/bash
# 部署模型到 Web 应用

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_DIR="$PROJECT_ROOT/web"
MODEL_DIR="$SCRIPT_DIR/models"
SRC_MODEL_DIR="$WEB_DIR/src/models"
QUANTIZATION_BYTES=""
SKIP_BUILD="false"
PYTHON_BIN="$(which python)"
PIP_CMD="$PYTHON_BIN -m pip"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quantization-bytes)
            QUANTIZATION_BYTES="$2"
            shift 2
            ;;
        --quantization-bytes=*)
            QUANTIZATION_BYTES="${1#*=}"
            shift
            ;;
        --skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: ./deploy_to_web.sh [--quantization-bytes 1|2|4] [--skip-build]"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "部署中文 BERT 模型到 Web 应用"
echo "========================================="
echo ""

# 检查 SavedModel
SAVED_MODEL_PATH="$MODEL_DIR/chinese_bert_model_savedmodel"
TFJS_OUTPUT_DIR="$MODEL_DIR/chinese_bert_model_js"

if [ ! -d "$SAVED_MODEL_PATH" ]; then
    echo "❌ 未找到 SavedModel: $SAVED_MODEL_PATH"
    echo ""
    echo "解决方案："
    echo "1. 重新运行训练脚本（会保存 SavedModel）:"
    echo "   python3 train_bert_tensorflow.py \\"
    echo "       --dataset datasets/dataset_merged.csv \\"
    echo "       --output models/chinese_bert_model.tflite"
    echo ""
    echo "2. 确保训练脚本已保存 SavedModel（训练时会自动保存）"
    echo ""
    exit 1
fi

echo "✅ 找到 SavedModel: $SAVED_MODEL_PATH"
echo ""

# 检查 tensorflowjs
echo "📦 检查 tensorflowjs..."
if ! "$PYTHON_BIN" -c "import tensorflowjs" 2>/dev/null; then
    echo "   正在安装 tensorflowjs..."
    $PIP_CMD install tensorflowjs --quiet
    echo "   ✅ tensorflowjs 安装完成"
else
    echo "   ✅ tensorflowjs 已安装"
fi

echo ""

# 转换模型
if [ -d "$TFJS_OUTPUT_DIR" ]; then
    echo "⚠️  检测到已存在的 TensorFlow.js 模型，将覆盖..."
    rm -rf "$TFJS_OUTPUT_DIR"
fi

echo "🔄 转换 SavedModel 为 TensorFlow.js 格式..."
echo "   （这可能需要几分钟）"

CONVERTER_FLAGS=(
    "--input_format=tf_saved_model"
    "--output_format=tfjs_graph_model"
    "--skip_op_check"
)

case "$QUANTIZATION_BYTES" in
    1)
        echo "   ➕ 启用 int8 量化 (--quantize_uint8)"
        CONVERTER_FLAGS+=("--quantize_uint8")
        ;;
    2)
        echo "   ➕ 启用 float16 量化 (--quantize_float16)"
        CONVERTER_FLAGS+=("--quantize_float16")
        ;;
    4)
        echo "   ➕ 启用 uint16 量化 (--quantize_uint16)"
        CONVERTER_FLAGS+=("--quantize_uint16")
        ;;
    "")
        echo "   ℹ️ 默认全精度；如需减小体积，可指定 --quantization-bytes 2（float16）或 1（int8）"
        ;;
    *)
        echo "   ⚠️ 不支持的 quantization-bytes=$QUANTIZATION_BYTES，将采用默认配置"
        ;;
esac

# Use `--` so argparse stops reading flags (e.g. --quantize_float16) and treats the paths as positional args.
"$PYTHON_BIN" -m tensorflowjs.converters.converter \
    "${CONVERTER_FLAGS[@]}" \
    -- \
    "$SAVED_MODEL_PATH" \
    "$TFJS_OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "   ❌ 转换失败"
    exit 1
fi

echo "   ✅ 转换完成"
echo ""

# 复制文件到 web/src/models（构建时会自动复制到 dist）
echo "📁 复制文件到 web/src/models 目录..."
mkdir -p "$SRC_MODEL_DIR"

# 复制 TensorFlow.js 模型
echo "   复制 TensorFlow.js 模型..."
cp -r "$TFJS_OUTPUT_DIR" "$SRC_MODEL_DIR/"

# 复制辅助文件
echo "   复制辅助文件..."
cp "$MODEL_DIR/chinese_bert_model_vocab.txt" "$SRC_MODEL_DIR/" 2>/dev/null || echo "   ⚠️  词汇表文件不存在"
cp "$MODEL_DIR/chinese_bert_model_labels.txt" "$SRC_MODEL_DIR/" 2>/dev/null || echo "   ⚠️  标签文件不存在"

echo "   ✅ 文件复制完成"
echo ""

# 构建项目
echo "🔨 构建 Web 应用..."
cd "$WEB_DIR"
if [ ! -d "node_modules" ]; then
    echo "   📦 检测到缺少依赖，正在安装 npm 依赖..."
    npm install
fi
if [ "$SKIP_BUILD" = "false" ]; then
npm run build
else
    echo "   ⏭️  跳过 npm run build（收到 --skip-build 参数）"
fi

echo ""
echo "========================================="
echo "✅ 部署完成！"
echo "========================================="
echo ""
echo "文件位置:"
echo "  - TensorFlow.js 模型: $SRC_MODEL_DIR/chinese_bert_model_js/"
echo "  - 词汇表: $SRC_MODEL_DIR/chinese_bert_model_vocab.txt"
echo "  - 标签文件: $SRC_MODEL_DIR/chinese_bert_model_labels.txt"
echo ""
echo "注意: 模型文件已复制到 web/src/models/，构建时会自动复制到 web/dist/models/"
echo ""
echo "下一步:"
echo "  1. 启动服务器: (cd web && npm run serve)"
echo "  2. 访问: http://localhost:8000/?model=chinese_tfjs"
echo ""
