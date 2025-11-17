/**
 * BERT 文本分类器
 * 使用 TensorFlow.js 加载 TFLite 模型并进行推理
 */

// 使用 CDN 导入 TensorFlow.js
declare const tf: any;

interface ClassificationResult {
  label: string;
  score: number;
}

interface BertClassifierConfig {
  modelPath: string;
  vocabPath: string;
  labelsPath: string;
  maxLength?: number;
}

export class BertClassifier {
  private model: any = null;
  private vocab: Map<string, number> = new Map();
  private labels: string[] = [];
  private maxLength: number = 128;
  private initialized: boolean = false;

  constructor(private config: BertClassifierConfig) {
    this.maxLength = config.maxLength || 128;
  }

  /**
   * 初始化分类器（加载模型、词汇表和标签）
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      console.log('📥 加载 BERT 分类器...');
      
      // 加载词汇表
      await this.loadVocab();
      
      // 加载标签
      await this.loadLabels();
      
      // 加载模型
      await this.loadModel();
      
      this.initialized = true;
      console.log('✅ BERT 分类器初始化完成');
    } catch (error) {
      console.error('❌ BERT 分类器初始化失败:', error);
      throw error;
    }
  }

  /**
   * 加载词汇表
   */
  private async loadVocab(): Promise<void> {
    console.log(`📖 加载词汇表: ${this.config.vocabPath}`);
    
    const response = await fetch(this.config.vocabPath);
    const text = await response.text();
    const lines = text.trim().split('\n');
    
    this.vocab.clear();
    lines.forEach((word, index) => {
      this.vocab.set(word.trim(), index);
    });
    
    console.log(`   ✅ 词汇表加载完成 (${this.vocab.size} 个词)`);
  }

  /**
   * 加载标签
   */
  private async loadLabels(): Promise<void> {
    console.log(`📖 加载标签: ${this.config.labelsPath}`);
    
    const response = await fetch(this.config.labelsPath);
    const text = await response.text();
    this.labels = text.trim().split('\n').filter(line => line.trim() !== '');
    
    console.log(`   ✅ 标签加载完成 (${this.labels.length} 个标签)`);
  }

  /**
   * 加载 TFLite 模型
   * 
   * 注意：TensorFlow.js 不能直接加载 TFLite 模型。
   * 需要先将 TFLite 模型转换为 TensorFlow.js 格式，或使用 TFLite Web API。
   * 
   * 转换方法：
   * 1. 使用 tensorflowjs_converter: 
   *    pip install tensorflowjs
   *    tensorflowjs_converter --input_format=tf_lite --output_format=tfjs_graph_model model.tflite model_js/
   * 
   * 2. 或者使用 TFLite Web API（如果浏览器支持）
   */
  private async loadModel(): Promise<void> {
    console.log(`📥 加载模型: ${this.config.modelPath}`);
    
    try {
      // 检查 TensorFlow.js 是否已加载
      if (typeof tf === 'undefined') {
        throw new Error('TensorFlow.js 未加载，请确保已引入 @tensorflow/tfjs');
      }

      // 方法 1: 尝试加载 TensorFlow.js 格式的模型（推荐）
      // modelPath 应该直接指向 model.json 文件
      let modelJsonPath = this.config.modelPath;
      
      // 如果路径是 .tflite，尝试转换为 TensorFlow.js 路径
      if (modelJsonPath.endsWith('.tflite')) {
        modelJsonPath = modelJsonPath.replace('.tflite', '_js/model.json');
      }
      // 如果路径是目录，添加 model.json
      else if (!modelJsonPath.endsWith('.json') && !modelJsonPath.endsWith('/')) {
        modelJsonPath = modelJsonPath + '/model.json';
      }
      // 如果路径以 / 结尾，添加 model.json
      else if (modelJsonPath.endsWith('/')) {
        modelJsonPath = modelJsonPath + 'model.json';
      }
      
      try {
        console.log(`   尝试加载模型: ${modelJsonPath}`);
        this.model = await tf.loadGraphModel(modelJsonPath);
        console.log('   ✅ 模型加载完成（TensorFlow.js 格式）');
        return;
      } catch (jsonError: any) {
        console.log(`   ⚠️  加载失败: ${jsonError?.message || String(jsonError)}`);
        console.log('   尝试其他方法...');
      }

      // 方法 2: 使用 TFLite Web API（如果可用）
      // 注意：这需要浏览器支持 WebAssembly 和 TFLite Web API
      if (typeof window !== 'undefined' && (window as any).tflite) {
        const tflite = (window as any).tflite;
        const modelResponse = await fetch(this.config.modelPath);
        const modelArrayBuffer = await modelResponse.arrayBuffer();
        this.model = await tflite.loadModel(modelArrayBuffer);
        console.log('   ✅ 模型加载完成（TFLite Web API）');
        return;
      }

      // 如果都失败，抛出错误
      throw new Error(
        '无法加载模型。请确保：\n' +
        '1. 模型已转换为 TensorFlow.js 格式（使用 tensorflowjs_converter）\n' +
        '2. 或使用支持 TFLite Web API 的浏览器'
      );
    } catch (error) {
      console.error('   ❌ 模型加载失败:', error);
      throw error;
    }
  }

  /**
   * BERT 分词（基于词汇表）
   */
  private tokenize(text: string): number[] {
    // 简单的字符级分词（适用于中文）
    // 对于更准确的分词，可以使用更复杂的算法
    const tokens: number[] = [];
    
    // 添加 [CLS] token
    const clsTokenId = this.vocab.get('[CLS]') ?? this.vocab.get('<s>') ?? 101;
    tokens.push(clsTokenId);
    
    // 处理文本
    // BERT 使用 WordPiece 分词，这里简化处理
    // 对于中文，可以按字符分割
    const chars = Array.from(text);
    for (const char of chars) {
      // 尝试直接匹配字符
      let tokenId = this.vocab.get(char);
      
      // 如果找不到，尝试查找子词
      if (tokenId === undefined) {
        // 简化处理：使用 UNK token
        tokenId = this.vocab.get('[UNK]') ?? this.vocab.get('<unk>') ?? 100;
      }
      
      tokens.push(tokenId);
    }
    
    // 添加 [SEP] token
    const sepTokenId = this.vocab.get('[SEP]') ?? this.vocab.get('</s>') ?? 102;
    tokens.push(sepTokenId);
    
    return tokens;
  }

  /**
   * 文本预处理
   */
  private preprocess(text: string): { inputIds: number[], attentionMask: number[] } {
    // 分词
    const tokens = this.tokenize(text);
    
    // 截断或填充到固定长度
    const inputIds: number[] = [];
    const attentionMask: number[] = [];
    
    for (let i = 0; i < this.maxLength; i++) {
      if (i < tokens.length) {
        inputIds.push(tokens[i]);
        attentionMask.push(1);
      } else {
        // 填充
        const padTokenId = this.vocab.get('[PAD]') ?? this.vocab.get('<pad>') ?? 0;
        inputIds.push(padTokenId);
        attentionMask.push(0);
      }
    }
    
    return { inputIds, attentionMask };
  }

  /**
   * 分类文本
   */
  async classify(text: string): Promise<ClassificationResult[]> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.model) {
      throw new Error('模型未加载');
    }

    try {
      // 预处理
      const { inputIds, attentionMask } = this.preprocess(text);
      
      // 转换为 TensorFlow.js 张量
      const inputIdsTensor = tf.tensor2d([inputIds], [1, this.maxLength], 'int32');
      const attentionMaskTensor = tf.tensor2d([attentionMask], [1, this.maxLength], 'int32');
      
      // 推理
      let predictions: any;
      
      // 根据模型输入格式调用
      if (this.model.inputs.length === 2) {
        // 两个输入：input_ids 和 attention_mask
        predictions = this.model.predict([inputIdsTensor, attentionMaskTensor]);
      } else {
        // 单个输入：input_ids
        predictions = this.model.predict(inputIdsTensor);
      }
      
      // 获取概率分布
      const probabilities = await predictions.data();
      
      // 清理张量
      inputIdsTensor.dispose();
      attentionMaskTensor.dispose();
      predictions.dispose();
      
      // 转换为结果格式
      const results: ClassificationResult[] = [];
      for (let i = 0; i < this.labels.length && i < probabilities.length; i++) {
        results.push({
          label: this.labels[i],
          score: probabilities[i]
        });
      }
      
      // 按分数排序
      results.sort((a, b) => b.score - a.score);
      
      return results;
    } catch (error) {
      console.error('分类失败:', error);
      throw error;
    }
  }

  /**
   * 清理资源
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.initialized = false;
  }
}

