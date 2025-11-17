/**
 * 模型配置文件
 * 
 * 用于管理不同模型的配置信息
 */

export interface ModelConfig {
  modelPath: string;
  displayName: string;
  description: string;
  language: string;
  usageHint: string;
  // TensorFlow.js 模型配置（可选）
  vocabPath?: string;
  labelsPath?: string;
  maxLength?: number;
  // 模型类型：'mediapipe' 或 'tensorflow'
  modelType?: 'mediapipe' | 'tensorflow';
}

export const MODEL_CONFIGS: Record<string, ModelConfig> = {
  // 默认英文模型
  english: {
    modelPath: 'https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/1/bert_classifier.tflite',
    displayName: 'English Sentiment Model (MediaPipe)',
    description: 'Pretrained MediaPipe text classifier optimized for English sentences. Best for English sentences. Chinese text may produce inaccurate predictions.',
    language: 'English',
    usageHint: 'Switch to this model when you want fast, lightweight English sentiment analysis directly in the browser.',
    modelType: 'mediapipe'
  },
  
  // 中文模型 - TensorFlow.js 版本（新版，推荐）
  chinese_tfjs: {
    modelPath: '/models/chinese_bert_model_js/model.json',
    vocabPath: '/models/chinese_bert_model_vocab.txt',
    labelsPath: '/models/chinese_bert_model_labels.txt',
    maxLength: 128,
    displayName: '中文情感分析模型 (TensorFlow.js)',
    description: '基于 BERT-Base-Chinese 微调的 TensorFlow.js 模型，适合中文情感分析与教学演示。',
    language: '中文',
    usageHint: '建议输入简体中文文本，模型将返回「积极 / 消极」的置信度。',
    modelType: 'tensorflow'
  }
};

export type ModelKey = keyof typeof MODEL_CONFIGS;

const DEFAULT_MODEL_KEY: ModelKey = 'chinese_tfjs';

// 当前使用的模型类型
export let CURRENT_MODEL_TYPE: ModelKey = DEFAULT_MODEL_KEY;

/**
 * 设置当前使用的模型类型
 */
export function setModelType(type: ModelKey): void {
  CURRENT_MODEL_TYPE = type;
  
  if (typeof window !== 'undefined' && window.localStorage) {
    window.localStorage.setItem('preferredModelType', type);
  }
}

/**
 * 获取当前模型的配置
 */
export function getCurrentModelConfig(): ModelConfig {
  return MODEL_CONFIGS[CURRENT_MODEL_TYPE];
}

/**
 * 获取初始模型（优先使用本地存储）
 */
export function getInitialModelType(): ModelKey {
  if (typeof window !== 'undefined' && window.localStorage) {
    const savedModel = window.localStorage.getItem('preferredModelType');
  if (savedModel && savedModel in MODEL_CONFIGS) {
      return savedModel as ModelKey;
  }
  }
  
  return DEFAULT_MODEL_KEY;
}

