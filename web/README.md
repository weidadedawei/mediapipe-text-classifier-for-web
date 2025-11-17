# Web 前端指南

基于 TypeScript、TensorFlow.js 与 Material Design Components 构建的中文情感分析 UI。该前端完全在浏览器中运行，通过加载 `training/` 输出的 TensorFlow.js 模型完成推理。

## 功能概览

- 浏览器端载入 BERT 中文情感模型，实时返回情感类别与置信度
- Material Design 风格的输入区与结果卡片
- 输入框支持回车（Enter）快捷键触发分类，兼顾键盘操作体验
- 纯静态部署：无需后端服务

## 目录结构

```
web/
├── src/            # TypeScript、样式、HTML、模型文件
├── dist/           # 构建输出
├── build.js        # 自定义构建脚本
├── package.json    # 前端依赖与 npm scripts
└── tsconfig.json
```

## 环境准备

```bash
cd web
npm install
```

Node.js 18+ 测试通过。若使用 pnpm / yarn，可自行映射脚本。

## 常用命令

| 命令 | 作用 |
|------|------|
| `npm run build` | 使用 `build.js` 生成产物到 `dist/` |
| `npm run serve` | 启动开发服务器（默认 8000 端口） |
| `npm run clean` | 清空 `dist/`（如在 `package.json` 中定义） |

运行 `npm run serve` 后访问 `http://localhost:8000`。

## 使用流程

1. **等待模型加载**：页面加载后会自动 fetch `src/models/<model>/model.json` 与 `.bin`。
2. **输入文本**：在文本域输入中文句子。
3. **点击「分类」或按下回车键**：触发推理，界面显示情感类别与置信度条形图。
4. **切换模型（可选）**：通过页面上的模型切换按钮在中文 TensorFlow.js 模型与英文 MediaPipe 模型之间切换。

## 模型文件放置约定

1. 确认在 `training/` 中通过 `deploy_to_web.sh` 或手动流程生成 TFJS 模型。
2. 将输出目录复制到 `web/src/models/<模型名>/`，典型结构：
   ```
   web/src/models/chinese_tfjs/
   ├── model.json
   ├── group1-shard1of3.bin
   ├── ...
   ├── chinese_bert_model_vocab.txt
   └── chinese_bert_model_labels.txt
   ```
3. 构建脚本会把整个 `src/models/` 拷贝到 `dist/models/`，保持相对路径一致。
4. 若新增模型，记得在前端配置（如 `src/config.ts`）中注册其标识，以便在 UI 中选择。

## 部署建议

- 构建产物为纯静态文件，可上传至任意静态托管（Vercel、Netlify、Nginx、S3 等）。
- 部署命令示例：
  ```bash
  cd web
  npm run build
  vercel --prod   # 依据项目记忆，采用本地 Vercel CLI
  ```
- 若放在自建服务器，确保配置 `dist/` 为根目录，并允许跨域获取模型文件。

## 浏览器要求

- 支持 ES6 模块、WebAssembly、Fetch API
- 已在以下版本验证：Chrome/Edge 88+、Firefox 89+、Safari 14.1+、Opera 74+
- 建议启用 WebGL 加速，以获得最佳 TensorFlow.js 性能

---

更多关于模型训练、部署流程与 FAQ，请参阅 `training/README.md`。

## Cloudflare Pages 部署

Cloudflare Pages 可免费托管静态站点，推理在浏览器端执行，不额外占用 Cloudflare CPU。

1. **安装 & 登录 Wrangler**
   ```bash
   npm install --global wrangler
   wrangler login
   ```
2. **构建前端**
   ```bash
   cd web
   npm install
   npm run build
   ```
3. **部署**
   ```bash
   npm run deploy:cf
   ```
   - 默认读取 `wrangler.toml` 中的 `project_name`
   - 输出路径 `web/dist`，等价于 `wrangler pages deploy dist`
4. **本地预览**
   ```bash
   wrangler pages dev dist
   ```
5. **环境变量**
   - 直接在 `wrangler.toml` 的 `[vars]` 中设置
   - 或在 Cloudflare 控制台 → Pages → Settings → Environment variables

> 参考：[Cloudflare Workers & Pages 官方文档](https://developers.cloudflare.com/workers/)

## 模型加载体验

- 页面在模型初始化期间会显示全屏动画及提示，防止用户误以为卡死。
- 未量化的中文 TensorFlow.js 模型约 **390 MB**。若希望缩短加载时间，可在 `training/deploy_to_web.sh` 中开启 `--quantization-bytes 2`（float16）或 `--quantization-bytes 1`（int8），然后重新构建 / 部署。
- 词表与标签文件仅在首次加载时下载，浏览器会自动缓存，可配合 Cloudflare Pages / CDN 进一步加速。
