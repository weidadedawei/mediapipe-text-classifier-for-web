/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// @ts-ignore - CDN import
import { MDCTextField } from "https://cdn.skypack.dev/@material/textfield";
// @ts-ignore - CDN import
import {
  TextClassifier,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@0.10.0";
import type { ModelConfig, ModelKey } from './config.js';
import { MODEL_CONFIGS, getCurrentModelConfig, getInitialModelType, setModelType } from './config.js';
import { BertClassifier } from './bert_classifier.js';

// TensorFlow.js 通过 <script> 标签加载，作为全局变量 tf
declare const tf: any;

// Type definitions for MediaPipe
interface TextClassifierResult {
  classifications: Array<{
    categories: Array<{
      categoryName: string;
      score: number;
    }>;
  }>;
}

// 中文标签映射
const labelTranslations: Record<string, string> = {
  'negative': '负面情绪',
  'positive': '正面情绪',
  'Negative': '负面情绪',
  'Positive': '正面情绪',
  'NEGATIVE': '负面情绪',
  'POSITIVE': '正面情绪'
};

// 获取中文标签
function getChineseLabel(englishLabel: string): string {
  return labelTranslations[englishLabel] || englishLabel;
}

interface ModelUICopy {
  documentTitle: string;
  heroBadge: string;
  heroTitle: string;
  heroSubtitle: string;
  modelSectionTitle: string;
  modelSectionSubtitle: string;
  modelButtonLabelChinese: string;
  modelButtonLabelEnglish: string;
  usageTitle: string;
  usageHint: string;
  inputLabel: string;
  fillSample: string;
  analyzeButton: string;
  resultTitle: string;
  loadingMessage: string;
  readyMessage: string;
  analyzingMessage: string;
  sampleText: string;
  mismatchWarning?: string;
  overlayHint?: string;
}

const CHINESE_SAMPLE = `今天天气真好，阳光明媚，微风习习。
我心情非常愉快，想要出去走走。
公园里的花儿开得正艳，鸟儿在枝头歌唱。
这样的日子让人感到无比幸福和满足。
让我们保持积极乐观的心态，享受每一天。`;

const ENGLISH_SAMPLE = `Today I woke up feeling energized and ready to start the day.
I am grateful for my teammates who always support each other.
Even when problems appear, I believe we can solve them with patience and empathy.`;

const UI_COPY: Record<ModelKey, ModelUICopy> = {
  chinese_tfjs: {
    documentTitle: '中文情感分析 Web 应用',
    heroBadge: '情感 AI 体验站',
    heroTitle: '中文情感分析演示',
    heroSubtitle: '完全在浏览器端运行的 TensorFlow.js 模型。中文情感分析模型 (TensorFlow.js)：基于 BERT-Base-Chinese 微调，适合中文情感分析与教学演示。建议输入简体中文文本，模型将返回「积极 / 消极 / 中性」的置信度。',
    modelSectionTitle: '模型选择',
    modelSectionSubtitle: '点击下方按钮即可在中文 TensorFlow.js 模型与英文 MediaPipe 模型之间切换。',
    modelButtonLabelChinese: '中文模型',
    modelButtonLabelEnglish: '英文模型',
    usageTitle: '如何使用',
    usageHint: '输入中文句子并点击按钮，立即查看情感分类结果。',
    inputLabel: '输入文本',
    fillSample: '填充示例文本',
    analyzeButton: '分类',
    resultTitle: '情感分析结果',
    loadingMessage: '正在加载中文 TensorFlow.js 模型，请稍候…',
    readyMessage: '✅ 中文模型已准备就绪，可直接输入中文句子进行分析。',
    analyzingMessage: '正在分类...',
    sampleText: CHINESE_SAMPLE,
    overlayHint: '模型约 390 MB，请在稳定网络环境下等待加载完成。'
  },
  english: {
    documentTitle: 'Sentiment Classifier Playground',
    heroBadge: 'Sentiment AI Playground',
    heroTitle: 'English Sentiment Demo',
    heroSubtitle: 'Classify English sentences instantly with the MediaPipe text classifier. English Sentiment Model (MediaPipe): Pretrained model optimized for English sentences; Chinese text may produce inaccurate predictions.',
    modelSectionTitle: 'Model selector',
    modelSectionSubtitle: 'Switch between the Chinese TensorFlow.js model and the English MediaPipe model.',
    modelButtonLabelChinese: 'Chinese Model',
    modelButtonLabelEnglish: 'English Model',
    usageTitle: 'How to use',
    usageHint: 'Type an English paragraph and click the button to see sentiment scores.',
    inputLabel: 'Input text',
    fillSample: 'Insert sample text',
    analyzeButton: 'Analyze',
    resultTitle: 'Sentiment result',
    loadingMessage: 'Loading the English MediaPipe model…',
    readyMessage: '✅ English model ready. Enter English sentences for the most accurate predictions.',
    analyzingMessage: 'Classifying...',
    sampleText: ENGLISH_SAMPLE,
    mismatchWarning: '⚠️ This model is optimized for English. Results for Chinese characters may be inaccurate.',
    overlayHint: 'MediaPipe English model体积较小，一般几秒即可完成。'
  }
};

// Get the required elements
let input: HTMLInputElement;
let output: HTMLElement;
let submit: HTMLButtonElement;
let defaultTextButton: HTMLButtonElement;
let demosSection: HTMLElement;
let textField: MDCTextField;
let heroTitle: HTMLElement | null;
let heroSubtitle: HTMLElement | null;
let heroBadge: HTMLElement | null;
let usageTitle: HTMLElement | null;
let usageHint: HTMLElement | null;
let inputLabel: HTMLElement | null;
let resultTitle: HTMLElement | null;
let populateTextLabel: HTMLElement | null;
let submitLabel: HTMLElement | null;
let modelLanguage: HTMLElement | null;
let modelName: HTMLElement | null;
let modelDescription: HTMLElement | null;
let modelHint: HTMLElement | null;
let loadingOverlay: HTMLElement | null;
let loadingOverlayText: HTMLElement | null;
let loadingOverlayHint: HTMLElement | null;
let modelSectionTitle: HTMLElement | null;
let modelSectionSubtitle: HTMLElement | null;
let modelSwitcher: HTMLElement | null;

let textClassifier: TextClassifier | null = null;
let bertClassifier: BertClassifier | null = null;
let currentModelRuntime: 'mediapipe' | 'tensorflow' | null = null;
let currentModelKey: ModelKey = getInitialModelType();
let isModelLoading = false;
const modelCache: Partial<Record<ModelKey, { runtime: 'mediapipe' | 'tensorflow'; classifier: TextClassifier | BertClassifier }>> = {};

// Initialize elements and event listeners when DOM is ready
function initializeApp() {
  // Get the required elements
  input = document.getElementById("input") as HTMLInputElement;
  output = document.getElementById("output") as HTMLElement;
  submit = document.getElementById("submit") as HTMLButtonElement;
  defaultTextButton = document.getElementById("populate-text") as HTMLButtonElement;
  demosSection = document.getElementById("demos") as HTMLElement;
  heroTitle = document.getElementById("hero-title");
  heroSubtitle = document.getElementById("hero-subtitle");
  heroBadge = document.getElementById("hero-badge");
  usageTitle = document.getElementById("usage-title");
  usageHint = document.getElementById("usage-hint");
  inputLabel = document.getElementById("input-label");
  resultTitle = document.getElementById("result-title");
  populateTextLabel = document.getElementById("populate-text-label");
  submitLabel = document.getElementById("submit-label");
  modelLanguage = document.getElementById("model-language");
  modelName = document.getElementById("model-name");
  modelDescription = document.getElementById("model-description");
  modelHint = document.getElementById("model-hint");
  loadingOverlay = document.getElementById("loading-overlay");
  loadingOverlayText = document.getElementById("loading-overlay-text");
  loadingOverlayHint = loadingOverlay?.querySelector("small") ?? null;
  modelSectionTitle = document.getElementById("model-section-title");
  modelSectionSubtitle = document.getElementById("model-section-subtitle");
  modelSwitcher = document.getElementById("model-switcher");

  // Initialize Material Design TextField
  const textFieldElement = document.querySelector(".mdc-text-field");
  if (textFieldElement) {
    textField = new MDCTextField(textFieldElement);
  }

  // Add a button click listener to add the default text
  if (defaultTextButton) {
    defaultTextButton.addEventListener("click", () => {
      if (input) {
        const copy = UI_COPY[currentModelKey];
        input.value = copy.sampleText;
      }
    });
  }
  
  // 允许使用回车键触发分类
  if (input) {
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        submit?.click();
      }
    });
  }

  // Add a button click listener that classifies text on click
  if (submit) {
    submit.addEventListener("click", async () => {
      const copy = UI_COPY[currentModelKey];
      
      if (isModelLoading) {
        showInfoBanner(copy.loadingMessage, 'info');
        return;
      }
      
      if (!input || input.value === "") {
        alert(currentModelKey === 'chinese_tfjs'
          ? "请输入一些文本，或点击“填充示例文本”按钮添加文本"
          : "Please enter some text or insert the sample paragraph.");
        return;
      }

      if (output) {
        showInfoBanner(copy.analyzingMessage, 'info');
      }

      await sleep(5);
      
      try {
        if (currentModelRuntime === 'tensorflow' && bertClassifier) {
          // 使用 TensorFlow.js 模型
          const results = await bertClassifier.classify(input.value);
          displayTensorFlowResults(results);
        } else if (currentModelRuntime === 'mediapipe' && textClassifier) {
          // 使用 MediaPipe 模型
          const result = textClassifier.classify(input.value);
          displayClassificationResult(result);
        } else {
          const fallbackMessage = currentModelKey === 'chinese_tfjs'
            ? "模型未加载，请稍候或尝试重新切换模型。"
            : "Model is not ready yet. Please wait or switch the model again.";
          showInfoBanner(fallbackMessage, 'warning');
        }
      } catch (error) {
        console.error('分类失败:', error);
        if (output) {
          const errorMessage = currentModelKey === 'chinese_tfjs'
            ? `分类失败: ${error}`
            : `Classification failed: ${error}`;
          output.innerText = errorMessage;
        }
      }
    });
  }

  // 绑定模型切换按钮
  if (modelSwitcher) {
    modelSwitcher.addEventListener("click", (event) => {
      const target = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-model-key]');
      if (!target) return;
      const modelKey = target.dataset.modelKey as ModelKey | undefined;
      if (!modelKey || modelKey === currentModelKey || isModelLoading) {
        return;
      }
      createTextClassifier(modelKey);
    });
  }

  // 初始化默认模型
  const initialModel = getInitialModelType();
  updateUICopy(initialModel);
  setActiveModelButton(initialModel);
  createTextClassifier(initialModel);
}

// Create the TextClassifier object upon page load
const createTextClassifier = async (modelKey?: ModelKey) => {
  if (isModelLoading) {
    return;
  }
  
  const targetKey = modelKey ?? getInitialModelType();
  const copy = UI_COPY[targetKey];
  const targetConfig = MODEL_CONFIGS[targetKey];
  
  updateUICopy(targetKey);
  updateModelInfoPanel(targetConfig);
  setActiveModelButton(targetKey);

  // 切换模型时清空输入与结果
  if (input) {
    input.value = "";
  }
  if (output) {
    output.innerHTML = "";
  }
  
  setModelType(targetKey);
  currentModelKey = targetKey;
  
  const cachedModel = modelCache[targetKey];
  if (cachedModel) {
    currentModelRuntime = cachedModel.runtime;
    if (cachedModel.runtime === 'tensorflow') {
      bertClassifier = cachedModel.classifier as BertClassifier;
      textClassifier = null;
    } else {
      textClassifier = cachedModel.classifier as TextClassifier;
      bertClassifier = null;
    }
    
    if (demosSection) {
      demosSection.classList.remove("invisible");
    }
    showInfoBanner(copy.readyMessage, 'success');
    hideLoadingOverlay();
    return;
  }
  
  showLoadingOverlay(copy.loadingMessage, copy.overlayHint);
  isModelLoading = true;
  
  try {
    const modelConfig = getCurrentModelConfig();
    
    console.log(`使用模型: ${modelConfig.displayName}`);
    console.log(`模型路径: ${modelConfig.modelPath}`);
    
    // 根据模型类型选择加载方式
    const modelType = modelConfig.modelType || 'mediapipe';
    currentModelRuntime = modelType;
    
    if (modelType === 'tensorflow') {
      // 使用 TensorFlow.js 模型
      if (!modelConfig.vocabPath || !modelConfig.labelsPath) {
        throw new Error('TensorFlow.js 模型需要 vocabPath 和 labelsPath 配置');
      }
      
      // 设置全局 tf 对象（如果未设置）
      if (typeof window !== 'undefined') {
        (window as any).tf = tf;
      }
      
      bertClassifier = new BertClassifier({
        modelPath: modelConfig.modelPath,
        vocabPath: modelConfig.vocabPath,
        labelsPath: modelConfig.labelsPath,
        maxLength: modelConfig.maxLength || 128
      });
      
      await bertClassifier.initialize();
      
      modelCache[targetKey] = {
        runtime: 'tensorflow',
        classifier: bertClassifier
      };
      
      console.log('✅ TensorFlow.js 模型加载完成');
    } else {
      // 使用 MediaPipe 模型（默认）
      const text = await FilesetResolver.forTextTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@0.10.0/wasm"
      );
      textClassifier = await TextClassifier.createFromOptions(text, {
        baseOptions: {
          modelAssetPath: modelConfig.modelPath
        },
        maxResults: 5
      });
      
      modelCache[targetKey] = {
        runtime: 'mediapipe',
        classifier: textClassifier
      };
      
      console.log('✅ MediaPipe 模型加载完成');
    }

    // Show demo section now model is ready to use.
    if (demosSection) {
      demosSection.classList.remove("invisible");
    }
    
    showInfoBanner(copy.readyMessage, 'success');
  } catch (error) {
    console.error("Failed to initialize TextClassifier:", error);
    if (output) {
      output.innerText = `模型加载失败: ${error}`;
    }
  } finally {
    isModelLoading = false;
    hideLoadingOverlay();
  }
};

// Initialize when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeApp);
} else {
  initializeApp();
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Iterate through the sentiment categories in the TextClassifierResult object, then display them in #output
function displayClassificationResult(result: TextClassifierResult) {
  if (!output) return;
  
  if (!result.classifications[0] || result.classifications[0].categories.length === 0) {
    const emptyMessage = currentModelKey === 'chinese_tfjs' ? "结果为空" : "No result";
    output.innerText = emptyMessage;
    return;
  }
  
    output.innerHTML = "";
    
    const inputText = input?.value || "";
    const hasChinese = /[\u4e00-\u9fa5]/.test(inputText);
  const copy = UI_COPY[currentModelKey];
    
  if (currentModelKey === 'english' && hasChinese && copy.mismatchWarning) {
      const warningDiv = document.createElement("div");
    warningDiv.className = "info-banner warning";
    warningDiv.innerText = copy.mismatchWarning;
      output.appendChild(warningDiv);
  }
  
  for (const category of result.classifications[0].categories) {
    const categoryDiv = document.createElement("div");
    const displayLabel = currentModelKey === 'english'
      ? `${category.categoryName} (${getChineseLabel(category.categoryName)})`
      : getChineseLabel(category.categoryName);
    const percentage = (category.score * 100).toFixed(1);
    categoryDiv.innerText = `${displayLabel}: ${percentage}%`;
    categoryDiv.style.marginBottom = "5px";
    
    if (category.score > 0.5) {
      categoryDiv.style.color = "#12b5cb";
      categoryDiv.style.fontWeight = "bold";
    }
    output.appendChild(categoryDiv);
  }
}

// Display TensorFlow.js classification results
function displayTensorFlowResults(results: Array<{label: string, score: number}>) {
  if (!output) return;
  
  if (results.length === 0) {
    const emptyMessage = currentModelKey === 'chinese_tfjs' ? "结果为空" : "No result";
    output.innerText = emptyMessage;
    return;
  }
  
  output.innerHTML = "";
  
  for (const result of results) {
    const categoryDiv = document.createElement("div");
    const percentage = (result.score * 100).toFixed(1);
    categoryDiv.innerText = `${result.label}: ${percentage}%`;
    categoryDiv.style.marginBottom = "5px";
    // highlight the likely category
    if (result.score > 0.5) {
      categoryDiv.style.color = "#12b5cb";
      categoryDiv.style.fontWeight = "bold";
    }
    output.appendChild(categoryDiv);
  }
}

async function disposeCurrentModels(): Promise<void> {
  if (textClassifier) {
    try {
      if ('close' in textClassifier && typeof (textClassifier as any).close === 'function') {
        await (textClassifier as any).close();
      }
    } catch (error) {
      console.warn('关闭 MediaPipe 模型失败：', error);
    } finally {
      textClassifier = null;
    }
  }
  
  if (bertClassifier) {
    bertClassifier.dispose();
    bertClassifier = null;
  }
}

function setActiveModelButton(modelKey: ModelKey): void {
  if (!modelSwitcher) return;
  const buttons = modelSwitcher.querySelectorAll<HTMLButtonElement>('button[data-model-key]');
  buttons.forEach((button) => {
    const isActive = button.dataset.modelKey === modelKey;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  });
}

function updateModelInfoPanel(config: ModelConfig): void {
  if (modelLanguage) {
    modelLanguage.innerText = config.language;
  }
  if (modelName) {
    modelName.innerText = config.displayName;
  }
  if (modelDescription) {
    modelDescription.innerText = config.description;
  }
  if (modelHint) {
    modelHint.innerText = config.usageHint;
  }
}

function updateUICopy(modelKey: ModelKey): void {
  const copy = UI_COPY[modelKey];
  document.title = copy.documentTitle;
  heroBadge && (heroBadge.innerText = copy.heroBadge);
  heroTitle && (heroTitle.innerText = copy.heroTitle);
  heroSubtitle && (heroSubtitle.innerText = copy.heroSubtitle);
  modelSectionTitle && (modelSectionTitle.innerText = copy.modelSectionTitle);
  modelSectionSubtitle && (modelSectionSubtitle.innerText = copy.modelSectionSubtitle);
  usageTitle && (usageTitle.innerText = copy.usageTitle);
  usageHint && (usageHint.innerText = copy.usageHint);
  inputLabel && (inputLabel.innerText = copy.inputLabel);
  resultTitle && (resultTitle.innerText = copy.resultTitle);
  populateTextLabel && (populateTextLabel.innerText = copy.fillSample);
  submitLabel && (submitLabel.innerText = copy.analyzeButton);
  
  if (loadingOverlayHint) {
    loadingOverlayHint.innerText = copy.overlayHint || '';
  }
  
  if (modelSwitcher) {
    const chineseButtonLabel = modelSwitcher.querySelector<HTMLSpanElement>('button[data-model-key="chinese_tfjs"] .model-name');
    const englishButtonLabel = modelSwitcher.querySelector<HTMLSpanElement>('button[data-model-key="english"] .model-name');
    chineseButtonLabel && (chineseButtonLabel.innerText = copy.modelButtonLabelChinese);
    englishButtonLabel && (englishButtonLabel.innerText = copy.modelButtonLabelEnglish);
  }
}

function showInfoBanner(message: string, variant: 'info' | 'success' | 'warning' = 'info'): void {
  if (!output) return;
  output.innerHTML = `<div class="info-banner ${variant}">${message}</div>`;
}

function showLoadingOverlay(message?: string, hint?: string): void {
  if (!loadingOverlay) return;
  loadingOverlay.classList.add('is-visible');
  if (loadingOverlayText && message) {
    loadingOverlayText.innerText = message;
  }
  if (loadingOverlayHint && hint) {
    loadingOverlayHint.innerText = hint;
  }
}

function hideLoadingOverlay(): void {
  if (!loadingOverlay) return;
  loadingOverlay.classList.remove('is-visible');
}
