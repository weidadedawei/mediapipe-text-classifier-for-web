# 中文情感分析 Web 应用

基于 TensorFlow.js 和 BERT 的 Web 文本分类应用，支持中文情感分析。完全在浏览器中运行，无需后端服务器。

## 功能特性

- 客户端运行：完全在浏览器中运行，无需后端服务器
- 中文情感分析：使用 BERT-Base-Chinese 模型对中文文本进行情感分类
- 实时分类：输入文本后即时获得分类结果
- Material Design：使用 Material Design Components 构建现代化 UI
- 结果可视化：显示分类类别和置信度分数

## 技术栈

- TypeScript：源代码使用 TypeScript 编写
- TensorFlow.js：在浏览器中运行机器学习模型
- BERT-Base-Chinese：用于中文文本分类的预训练模型
- Material Design Components：用于 UI 组件
- esbuild：用于快速打包前端代码和依赖

## 项目结构

```
mediapipe-text-classifier-for-web/
├── web/                    # 前端代码（TensorFlow.js）
│   ├── src/                # TypeScript、HTML、样式与模型文件
│   ├── dist/               # 构建输出目录
│   ├── package.json        # 前端依赖与脚本
│   ├── build.js            # 构建脚本
│   └── tsconfig.json       # TypeScript 配置
├── training/               # 训练与部署脚本
│   ├── models/             # 训练好的模型
│   ├── train_bert_tensorflow.py
│   ├── deploy_to_web.sh
│   ├── requirements.txt
│   └── setup_env.sh
└── README.md               # 项目说明文档
```

## 子项目概览

- `web/`：前端（TypeScript + TensorFlow.js），负责在浏览器端载入模型、执行推理，并提供 Material Design 风格的交互界面。详见 `web/README.md`。
- `training/`：训练、转换与部署脚本，涵盖数据准备、BERT 微调、TFLite/TensorFlow.js 导出以及自动部署流程。详见 `training/README.md`。

## 快速上手总览

1. **训练模型**
   - 在 `training/` 目录中准备 Conda/venv 环境、清洗数据集、运行 `train_bert_tensorflow.py`。
   - 使用 `deploy_to_web.sh` 或手动流程将模型转换成 TensorFlow.js 并复制到 `web/src/models/`。
   - 参考 `training/README.md` 获取完整参数说明、FAQ 与性能优化建议。
2. **运行前端**
   - 进入 `web/`，执行 `npm install && npm run build`。
   - `npm run build` 会使用 esbuild 打包所有依赖到 `dist/bundle.js`。
   - 使用 `npm run serve` 启动开发服务器，访问 `http://localhost:8000` 体验。

## 文档索引

- `training/README.md`：训练脚本、环境准备、数据规范、部署流程与常见问题。
- `web/README.md`：前端项目结构、依赖安装、开发/构建命令、模型放置约定与浏览器兼容性。
- `wrangler.toml` + “Cloudflare Pages 部署”章节：使用 Wrangler 直接部署到 Cloudflare Pages。

## Cloudflare Pages 部署

项目为纯静态前端，加载模型与推理均在浏览器端完成，可直接托管在 Cloudflare Pages（免费额度内无需额外 CPU 费用）。

### 依赖准备

```bash
npm install --global wrangler
wrangler login    # 浏览器授权 Cloudflare 账号
```

仓库已包含示例 `wrangler.toml`，其中：
- `pages.project_name`：Cloudflare Pages 项目名（可在第一次部署时创建）
- `build_command`：`npm install && npm run build`
- `build_output_dir`：`web/dist`

### 构建与部署

```bash
cd web
npm install
npm run build

# 部署到 Cloudflare Pages
npm run deploy:cf
```

脚本等价于 `wrangler pages deploy dist --project-name=mediapipe-text-classifier-for-web`，默认从 `web/dist/` 读取静态资源。

### 本地验证

```bash
wrangler pages dev web/dist
```

可在本地模拟 Cloudflare Pages 运行环境，验证模型加载和推理逻辑。

### 常用操作

- **查看部署历史**：Cloudflare Dashboard → Pages → 选中项目
- **回滚版本**：在 Pages 项目里选中历史部署并点击 “Rollback”
- **环境变量**：编辑 `wrangler.toml` 的 `[vars]` 段或在 Cloudflare 控制台设置，将在前端运行时注入（如 API endpoint）

### 常见部署/加载问题

| 问题 | 处理 |
|------|------|
| `wrangler pages deploy` 提示 “Need to specify how to reconcile divergent branches” 或 “Configuration file cannot contain both main and pages_build_output_dir” | 使用 `npm run deploy:cf`（已固定 `--branch=production`），并保持 `wrangler.toml` 仅包含 Pages 所需字段（`name`、`pages_build_output_dir`、`build_command`、`[vars]` 等）。 |
| 浏览器加载中文模型时卡住 | 未量化模型约 390 MB。请运行 `./training/deploy_to_web.sh --quantization-bytes 2`（float16）后重新构建；网页也会显示 Loading Overlay，请耐心等待。 |
| 推理时报 `Unknown op 'Erfc'` | 需要浏览器端 `@tensorflow/tfjs` ≥ 4.18 且使用 `webgl` backend，前端已在 `BertClassifier` 中注册自定义 `Erfc` 算子。如仍报错，清除缓存并确认 `tf.version_core` 与 `tf.getBackend()`。 |
| `deploy_to_web.sh` 转换失败，报 NumPy 2.x | 脚本已改用当前虚拟环境的 `python -m pip`。请确保在激活 env 后运行，并保持 `numpy<2`（建议 1.26.4）。详见 `training/README.md`。 |

## 许可证

本项目采用 Apache License 2.0 许可证。

## 相关资源

- [TensorFlow 官方文档](https://www.tensorflow.org/)
- [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers)
- [TensorFlow.js 文档](https://www.tensorflow.org/js)
- [BERT-Base-Chinese 模型](https://huggingface.co/bert-base-chinese)
