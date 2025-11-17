# TensorFlow BERT 训练脚本说明

下面的内容基于 `training/train_bert_tensorflow.py` 的现有实现做了校准，方便在回到代码时快速对照。

---

## 🚀 快速上手（最短路径）
如果你第一次接触 BERT 训练，可以照着下面的“菜谱”一步步照做，确保每一步都能得到即时反馈：

1. **准备数据**  
   把带有 `text`、`label` 两列的 CSV 放在 `training/datasets/` 下，例如 `datasets/dataset_merged.csv`。`label` 只允许出现“积极”“消极”这类文字标签。
2. **创建隔离环境（推荐）**
   ```bash
   cd training
   python -m venv .venv        # 或运行 setup_env.sh
   source .venv/bin/activate
   pip install -r requirements.txt
   # Apple 芯片建议再装
   pip install tensorflow-macos tensorflow-metal
   ```
3. **运行训练脚本**
   ```bash
   python train_bert_tensorflow.py \
       --dataset datasets/dataset_merged.csv \
       --output models/chinese_bert_model.tflite
   ```
   终端会实时打印数据量、GPU 使用情况、每个 epoch 的 loss/accuracy，以及最终生成的模型/日志路径。
4. **更新参数**  
   想要跑更多轮或更长句子？只需追加 `--epochs 5`、`--max-length 256` 等参数，所有可选项在文件底部的 `argparse` 定义里都能找到。

只要上面四步能顺利跑通，你就具备了调整和扩展脚本的基础。

---

## 🧩 环境准备（Conda / 脚本二选一）

### Conda 路线
1. 安装 [Anaconda](https://www.anaconda.com/download) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。
2. 在项目根目录执行：
   ```bash
   cd training
   conda create -n bert_gpu_env python=3.11 -y
   conda activate bert_gpu_env
   python -m pip install -r requirements.txt
   ```
3. Apple Silicon 建议再安装 `tensorflow-macos tensorflow-metal` 以启用 Metal GPU。

### 脚本捷径

```bash
cd training
./setup_env.sh
conda activate bert_gpu_env   # 或根据脚本输出选择 env
```

### 安装验证

```bash
python -c "import tensorflow as tf; import transformers; import numpy as np; print('TensorFlow:', tf.__version__); print('Transformers:', transformers.__version__); print('NumPy:', np.__version__)"
```

> `requirements.txt` 已固定 `transformers>=4.30,<4.50` 与 `numpy<2.0`，可避免 `builtins.safe_open` 与 NumPy 2.x 兼容性问题。

---

## 📂 数据集准备

- **格式**：CSV，至少包含 `text`（中文文本）与 `label`（情感标签）两列，编码 UTF-8。
- **示例**：

| 列名 | 说明 | 示例 |
|------|------|------|
| `text` | 中文文本内容 | 今天天气真好，心情愉快 |
| `label` | 情感标签 | 积极 |

- 仓库提供 `datasets/dataset_merged.csv` 作为模板，可直接复制再替换数据。

---

## ⚙️ 训练命令速查

### 基础命令

```bash
cd training
conda activate bert_gpu_env

python train_bert_tensorflow.py \
    --dataset datasets/dataset_merged.csv \
    --output models/chinese_bert_model.tflite
```

### 自定义参数

```bash
python train_bert_tensorflow.py \
    --dataset datasets/dataset_merged.csv \
    --output models/chinese_bert_model.tflite \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --max-length 128
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 训练数据 CSV 路径 | 必填 |
| `--output` | 导出 TFLite 路径 | `models/chinese_bert_model.tflite` |
| `--epochs` | 训练轮数 | 3 |
| `--batch-size` | 批次大小 | 16 |
| `--learning-rate` | 学习率 | 2e-5 |
| `--validation-split` | 验证集比例 | 0.2 |
| `--max-length` | 最大序列长度 | 128 |
| `--model-name` | 预训练模型 | `bert-base-chinese` |

---

## 💬 第一部分：底层原理（脚本究竟做了什么？）
把模型当成一位“博士”：Google 事先用海量语料训练好 BERT（例如 `bert-base-chinese`），他已经掌握语言，但还没做过“情感判断”这份具体工作。脚本做的就是把这位博士请来，再拿我们自己的数据集做一次“上岗培训”——这就是 **微调 (fine-tuning)**。

1. **载入预训练大脑**：`build_model()` 会通过 `TFBertForSequenceClassification.from_pretrained()` 把 Hugging Face 仓库里的模型和权重下载到本地。模型里包含 1 亿多参数，这些都是博士早已具备的语义知识。
2. **追加一个轻量分类头**：`TFBertForSequenceClassification` 会自动接上一个全连接层用于两分类（积极/消极）。训练时主要是调整这个“小判断器”，并对 BERT 的部分参数做轻微微调。
3. **喂入自己的样本重新训练**：脚本把 `datasets/dataset_merged.csv` 里的 12 万多条语料拆成训练/验证集，通过 `model.fit()` 迭代 3 个 epoch；所有梯度计算都交给 Apple Silicon 的 GPU (`tensorflow-metal`) 来加速。
4. **导出轻量模型**：训练完后还会把模型导出为 `.tflite`、SavedModel、词表和标签文件，方便部署到 Web/移动端。

---

## 🛠 第二部分：依赖的作用（为什么需要这些包？）
脚本依赖的包都列在 `training/requirements.txt`，其中每一项和代码都有直接对应关系：

- **Python 环境 / Conda (可选)**：建议用 `setup_env.sh` 创建一个隔离的虚拟环境，避免和系统 Python 或 `pyenv` 冲突。
- **`tensorflow` 2.13+**：运行神经网络的框架。若在 Apple Silicon 上训练，需要安装 `tensorflow-macos` 与 `tensorflow-metal` 插件并通过同样的 `import tensorflow as tf` 使用，它会让脚本自动识别 Metal GPU (`device: GPU:0, name: METAL`)。
- **`transformers` + `tokenizers`**：提供 `TFBertForSequenceClassification` 与 `BertTokenizer`。前者就是“博士模型”，后者是把中文切成 BERT 需要的词 ID 的“字典”。
- **`pandas`**：`load_dataset()` 用它读取/清洗 CSV，并统计标签分布。
- **`numpy`**：在验证阶段做 `np.argmax` 等基础张量操作。
- **`scikit-learn`**：提供 `train_test_split`（拆训练/验证集）与 `classification_report`（生成成绩单）。

安装方式：
```bash
cd training
pip install -r requirements.txt
# Apple Silicon 额外推荐
pip install tensorflow-macos tensorflow-metal
```

---

## 📜 第三部分：源码导览（如何与脚本对照？）
`train_bert_tensorflow.py` 中的主要模块如下，可直接跳到对应函数阅读：

1. **`load_dataset()`（第 31 行起）**：
   - 用 `pandas.read_csv` 读取 `text`、`label` 列，剔除空值。
   - 统计总量与标签分布，并生成 `label_to_id`、`id_to_label` 映射，供后续编码与评估使用。

2. **`prepare_data()`（第 60 行起）**：
   - 使用 `BertTokenizer` 把文本转为 `input_ids`、`attention_mask`，并固定到 `max_length`（默认 128）。
   - 返回 `tf.data.Dataset`，后面会 `shuffle`、`batch`、`prefetch` 以匹配模型输入。

3. **`build_model()`（第 87 行起）**：
   - 创建 `BertConfig(num_labels=<标签数>)`，然后构造 `TFBertForSequenceClassification`。
   - 预训练权重会自动加载，脚本同时打印参数量帮助快速确认是否成功。

4. **`train_model()`（第 105 行起）**：这是主装配线，`main()` 解析 CLI 参数后也会调用它。
   - **数据准备**：调用上述函数得到 `train_dataset`、`val_dataset`，并进行 `shuffle(1000).batch(batch_size)`。
   - **优化器选择**：脚本检测 Apple Silicon (`platform.machine() == 'arm64'`)，若成立则改用 `tf.keras.optimizers.legacy.Adam`，这是目前在 Metal 上最稳定、最快的实现。
   - **训练与回调**：`model.fit` 中启用了 `EarlyStopping`、`ReduceLROnPlateau`，避免过拟合并自动调整学习率。
   - **评估与报告**：用 `model.predict` 得到验证集 logits，经 `classification_report` 输出精确率/召回率，并把结果写入 `_evaluation.txt`。
   - **导出**：
     - 首先尝试通过 `tf.lite.TFLiteConverter.from_concrete_functions` 导出 `.tflite`；若失败自动回退到 SavedModel 流程。
     - 不论 TFLite 是否成功，都会另外保存 SavedModel（便于 TensorFlow.js 转换）、词表、标签列表和 JSON 映射。
   - **日志**：生成 `_training_log.txt`，记录参数、数据规模和最终指标。

5. **`main()`（文件底部）**：
   - 负责解析诸如 `--dataset`、`--output`、`--epochs` 等 CLI 入参，然后把它们传进 `train_model()`。

借助这份说明，你可以把 README 与源文件交叉验证：每个步骤在代码里都有对应实现，便于初学者对照学习而不是仅仅“跑通脚本”。

---

## 📦 训练输出

运行成功后，`training/models/` 下会出现：

```
├── chinese_bert_model.tflite
├── chinese_bert_model_savedmodel/
├── chinese_bert_model_vocab.txt
├── chinese_bert_model_labels.txt
├── chinese_bert_model_label_map.json
├── chinese_bert_model_evaluation.txt
└── chinese_bert_model_training_log.txt
```

- `.tflite`：移动端/浏览器 WASM 推理的精简模型。
- `SavedModel`：用于 TensorFlow.js 转换或进一步微调。
- `*_vocab.txt` / `*_labels.txt` / `*_label_map.json`：推理所需的 tokenizer 与标签映射。
- `*_evaluation.txt` / `*_training_log.txt`：方便追溯训练过程。

---

## ⚡️ 性能优化提示

- **Apple Silicon (M1/M2/M3)**：脚本自动切换到 `tf.keras.optimizers.legacy.Adam` 并启用 `tensorflow-metal`，可获得约 10× 提速。
- **内存不足时**：尝试 `--batch-size 8` 或更小，或把 `--max-length` 降到 64。
- **模型更快部署**：必要时可改用更小的 `--model-name`（如 `uer/roberta-base-finetuned-chinanews-chinese`）或开启量化（参考 TensorFlow Lite 指南）。

---

## 🚀 部署模型到 Web

### 方法一：自动脚本（推荐）

```bash
cd training
conda activate bert_gpu_env
./deploy_to_web.sh
```

脚本流程：检测 SavedModel → 转成 TensorFlow.js → 复制到 `web/src/models/` → 触发 `npm run build`。

#### 可选：量化/压缩输出

浏览器需要顺序下载约 **390 MB** 的中文模型（98 个 `.bin` 分片）。为减少加载耗时，可在转换阶段开启 `quantization_bytes`：

```bash
# float16（推荐，几乎无感知精度损失）
./deploy_to_web.sh --quantization-bytes 2

# int8（体积最小，但需验证精度）
./deploy_to_web.sh --quantization-bytes 1
```

`tensorflowjs_converter` 会自动将权重压缩为 16-bit 或 8-bit，常见节省比例 40%~70%。脚本会在控制台提示具体配置，也支持 `--skip-build`（仅复制模型，跳过 `npm run build`）。

### 方法二：手动控制

1. **转换为 TensorFlow.js**
   ```bash
   pip install tensorflowjs
   tensorflowjs_converter \
       --input_format=tf_saved_model \
       --output_format=tfjs_graph_model \
       models/chinese_bert_model_savedmodel \
       models/chinese_bert_model_js/
   ```
2. **复制到前端**
   ```bash
   mkdir -p ../web/src/models
   cp -r models/chinese_bert_model_js ../web/src/models/
   cp models/chinese_bert_model_vocab.txt ../web/src/models/
   cp models/chinese_bert_model_labels.txt ../web/src/models/
   ```
3. **构建前端**
   ```bash
   cd ../web
   npm install
   npm run build   # 或 npm run serve
   ```

访问 `http://localhost:8000/?model=chinese_tfjs` 即可验证。

---

## 🎨 故事版理解（让比喻带你读懂代码）
如果你更喜欢“讲故事”的解法，下面这份逐段拆解能帮你在脑海里构建完整流程。

### 💬 我们在做什么？
想象你要教一个小孩（神经网络）判断句子情绪：

1. **传统“笨方法”**  
   从零开始教他认识所有汉字，再慢慢理解语法，需要巨量数据与费用，几乎不可行。
2. **现代“聪明方法”**  
   直接请一位“已经读完博士”的语言学家——BERT。Google 已经为它付过昂贵的学费，它懂语法、上下文、常识。我们要做的只是微调，让它学会“情感分析”这份工作。

微调时我们在 BERT 的“大脑”后面挂一个小小的“判断器”（脚本里的分类层），然后用自己的 12 万条样本教它“积极/消极”的辨别方式。因为博士本身就很聪明，所以：训练快、成本低、准确率高。

### 🛠 依赖的作用
把项目想成“机器人制造工厂”：

- `conda` / 虚拟环境：搭建无尘车间，防止系统 Python、pyenv 等环境污染。
- `python` 3.11：整个工厂的操作系统。
- `tensorflow-macos` + `tensorflow-metal` + `keras`：机器人的“大脑”与控制面板，也是让 GPU（Metal）跑得飞快的关键。
- `transformers`：预制件仓库，内含 BERT 模型与 `BertTokenizer`，省得我们重造轮子。
- `tokenizers`、`safetensors`：仓库配套工具，负责快速分词与安全加载 400MB 级别的权重。
- `pandas`：CSV 录入员。
- `scikit-learn`：负责拆分训练/验证集并生成分类报告的质检员。

### 📜 源码工作流
从 `main()` 往上看，就像流水线一样：

1. `main()`：解析命令行参数，交给 `train_model()`。
2. `train_model()`：总装配线，从加载数据→构造 tokenizer→拆分数据→打包成 tf.data→构建模型→选择优化器（检测到 arm64 就用 `legacy.Adam`）→`model.fit()` 训练→`model.predict()` 评估→TFLite/SavedModel 导出→保存词表、标签、日志。
3. `load_dataset()`：pandas 读取 CSV，清洗空行，生成标签映射。
4. `prepare_data()`：BertTokenizer 把中文句子翻译成 ID/Mask，并封装成 `tf.data.Dataset`。
5. `build_model()`：从 Hugging Face 拉取 `TFBertForSequenceClassification` 并设置分类数量。

这样一来，你既能用“工程视角”理解脚本，又能用“故事版”在脑中快速复现整个流程。

---

## ❓ 常见问题

1. **训练时内存不足**：减小 `--batch-size` 或 `--max-length`，必要时切换更小模型。
2. **TensorFlow.js 加载失败**：确认已转换为 TFJS、路径正确，浏览器控制台没有 404，`.json` 与 `.bin` 文件齐全。
3. **推理速度慢**：使用量化模型、降低 `max_length`，或确保浏览器启用 WebGL/WASM。
4. **准确率不高**：增加数据量/epoch、调整学习率、检查数据质量。
5. **NumPy / safe_open 报错**：删除旧虚拟环境，重新使用 `python -m pip install -r requirements.txt`，保持 `numpy<2.0` 与 `transformers<4.50`。
