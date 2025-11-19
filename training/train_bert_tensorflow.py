#!/usr/bin/env python3
"""
TensorFlow BERT Chinese Sentiment Analysis Training Script

This script fine-tunes a pre-trained BERT model (e.g., bert-base-chinese) for sentiment classification.
It uses TensorFlow and Hugging Face Transformers.

Features:
- Loads and validates CSV datasets.
- Fine-tunes BERT for binary classification (positive/negative).
- Exports models to TFLite (for mobile/web) and SavedModel (for TF.js).
- Generates training logs and evaluation reports.

Usage:
    python3 train_bert_tensorflow.py --dataset dataset.csv --output models/chinese_bert_model.tflite
"""

import argparse
import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    import tensorflow as tf
    from transformers import (
        TFBertForSequenceClassification,
        BertTokenizer,
        BertConfig
    )
except ImportError as e:
    print("❌ Error: Missing required dependencies.")
    print(f"   Details: {str(e)}")
    print("   Please run: pip install tensorflow transformers pandas scikit-learn numpy")
    sys.exit(1)


def load_dataset(dataset_path):
    """加载 CSV 数据集"""
    print(f"📖 加载数据集: {dataset_path}")
    
    df = pd.read_csv(dataset_path, encoding='utf-8')
    
    # 检查必要的列
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"数据集必须包含 'text' 和 'label' 列。当前列: {list(df.columns)}")
    
    # 检查是否为空
    if len(df) == 0:
        raise ValueError("数据集为空")

    # 清理数据
    original_len = len(df)
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].astype(str).str.strip() != '']
    
    if len(df) == 0:
        raise ValueError("清理后的数据集为空（所有行都包含空值或空文本）")
        
    if len(df) < original_len:
        print(f"   ⚠️  已移除 {original_len - len(df)} 条无效数据（空值或空文本）")
    
    # 获取标签映射
    unique_labels = sorted(df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"   ✅ 数据集加载成功")
    print(f"   数据量: {len(df)} 条")
    print(f"   标签: {unique_labels}")
    print(f"   标签分布:")
    for label, count in df['label'].value_counts().items():
        print(f"     {label}: {count} 条")
    
    return df, label_to_id, id_to_label


def prepare_data(df, tokenizer, label_to_id, max_length=128):
    """准备训练数据"""
    print(f"\n📦 准备训练数据（最大长度: {max_length}）...")
    
    texts = df['text'].tolist()
    labels = [label_to_id[label] for label in df['label'].tolist()]
    
    # 使用 tokenizer 编码文本
    print("   正在编码文本...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    
    # 转换为 TensorFlow 数据集
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        },
        tf.constant(labels, dtype=tf.int32)
    ))
    
    print(f"   ✅ 数据准备完成")
    return dataset


def build_model(num_labels, model_name='uer/roberta-small-wwm-chinese-cluecorpussmall'):
    """构建 BERT 分类模型"""
    print(f"\n⚙️  构建模型: {model_name}")
    
    try:
        # 加载预训练的 BERT 模型
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_act="gelu"  # 强制使用兼容的 GELU 版本
        )
        model = TFBertForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
    except Exception as e:
        raise RuntimeError(f"无法加载预训练模型 '{model_name}': {str(e)}\n请检查网络连接或模型名称是否正确。")
    
    print(f"   ✅ 模型构建完成")
    print(f"   参数量: {model.count_params():,}")
    
    return model


def train_model(
    dataset_path,
    output_path,
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    validation_split=0.2,
    max_length=128,
    model_name='bert-base-chinese'
):
    """训练中文情感分析模型"""
    
    print("=" * 70)
    print("TensorFlow BERT - 中文情感分析模型训练")
    print("=" * 70)
    print()
    
    # 检查数据集文件
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 找不到数据集文件: {dataset_path}")
        return False
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")
    
    try:
        # 步骤 1: 加载数据集
        df, label_to_id, id_to_label = load_dataset(dataset_path)
        
        # 步骤 2: 加载 tokenizer
        print(f"\n📥 加载 BERT Tokenizer: {model_name}")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print(f"   ✅ Tokenizer 加载成功")
        
        # 步骤 3: 划分数据集
        print(f"\n📦 划分数据集（验证集比例: {validation_split}）...")
        train_df, val_df = train_test_split(
            df,
            test_size=validation_split,
            random_state=42,
            stratify=df['label']
        )
        print(f"   ✅ 训练集: {len(train_df)} 条")
        print(f"   ✅ 验证集: {len(val_df)} 条")
        
        # 步骤 4: 准备数据
        train_dataset = prepare_data(train_df, tokenizer, label_to_id, max_length)
        val_dataset = prepare_data(val_df, tokenizer, label_to_id, max_length)
        
        # 批处理
        train_dataset = train_dataset.shuffle(1000).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        
        # 步骤 5: 构建模型
        num_labels = len(label_to_id)
        model = build_model(num_labels, model_name)
        
        # 步骤 6: 编译模型
        print(f"\n🔧 编译模型...")
        
        # 检测 Apple Silicon (M1/M2/M3) Mac，使用 legacy 优化器以获得更好的性能
        import platform
        is_apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'
        
        if is_apple_silicon:
            # Apple Silicon Mac: 使用 legacy 优化器（速度提升 10 倍）
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            print(f"   优化器: Adam (legacy, 针对 Apple Silicon 优化)")
        else:
            # 其他平台: 使用标准优化器
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            print(f"   优化器: Adam (标准)")
        
        print(f"   学习率: {learning_rate}")
        print(f"   损失函数: SparseCategoricalCrossentropy")
        
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # 步骤 7: 训练模型
        print(f"\n🚀 开始训练模型...")
        print(f"   训练轮数: {epochs}")
        print(f"   批次大小: {batch_size}")
        print(f"   （这可能需要较长时间，取决于数据量和硬件）")
        print()
        
        # 回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1,
                min_lr=1e-6
            )
        ]
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("   ✅ 模型训练完成！")
        
        # 步骤 8: 评估模型
        print(f"\n📊 评估模型性能...")
        val_predictions = model.predict(val_dataset)
        val_pred_labels = np.argmax(val_predictions.logits, axis=1)
        val_true_labels = np.array([label_to_id[label] for label in val_df['label'].tolist()])
        
        accuracy = accuracy_score(val_true_labels, val_pred_labels)
        print(f"   验证集准确率: {accuracy:.4f}")
        
        # 分类报告
        print(f"\n   分类报告:")
        report = classification_report(
            val_true_labels,
            val_pred_labels,
            target_names=[id_to_label[i] for i in range(num_labels)],
            digits=4
        )
        print(report)
        
        # 保存评估报告
        report_path = output_path.replace('.tflite', '_evaluation.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("模型评估报告\n")
            f.write("=" * 70 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"数据集: {dataset_path}\n")
            f.write(f"训练集大小: {len(train_df)} 条\n")
            f.write(f"验证集大小: {len(val_df)} 条\n\n")
            f.write(f"验证集准确率: {accuracy:.4f}\n\n")
            f.write("分类报告:\n")
            f.write(report)
        
        print(f"   💾 评估报告已保存: {report_path}")
        
        # 步骤 9: 导出 TFLite 模型
        print(f"\n💾 导出 TFLite 模型...")
        print(f"   输出路径: {output_path}")
        
        tflite_success = False
        tflite_model = None
        
        # 创建包装模型用于 TFLite 转换
        # 注意：TFLite 转换需要 Keras 模型，但 Transformers 模型是函数式 API
        # 我们需要创建一个包装模型
        class TFLiteModel(tf.keras.Model):
            def __init__(self, bert_model):
                super().__init__()
                self.bert_model = bert_model
            
            # 注意：input_signature 必须与 call 方法的参数签名完全匹配
            def call(self, input_ids, attention_mask):
                outputs = self.bert_model({'input_ids': input_ids, 'attention_mask': attention_mask})
                return tf.nn.softmax(outputs.logits)
        
        # 创建包装模型
        tflite_wrapper = TFLiteModel(model)
        
        # 定义输入签名（用于 TFLite 转换）
        input_signature = [
            tf.TensorSpec(shape=[None, max_length], dtype=tf.int32, name='input_ids'),
            tf.TensorSpec(shape=[None, max_length], dtype=tf.int32, name='attention_mask')
        ]
        
        # 创建带签名的推理函数
        @tf.function(input_signature=input_signature)
        def model_inference(input_ids, attention_mask):
            return tflite_wrapper(input_ids, attention_mask)
        
        # 测试模型
        test_input_ids = tf.zeros((1, max_length), dtype=tf.int32)
        test_attention_mask = tf.ones((1, max_length), dtype=tf.int32)
        _ = model_inference(test_input_ids, test_attention_mask)
        
        # 转换为 TFLite（使用 concrete function）
        try:
            concrete_func = model_inference.get_concrete_function()
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            tflite_success = True
        except Exception as e1:
            # 如果第一种方法失败，尝试使用 SavedModel 方式
            print(f"   ⚠️  方法 1 失败: {str(e1)}")
            print(f"   尝试方法 2: 使用 SavedModel 转换...")
            
            try:
                # 保存为 SavedModel
                saved_model_path = output_path.replace('.tflite', '_savedmodel')
                # 创建一个接受字典输入的包装函数
                @tf.function(input_signature=[{
                    'input_ids': tf.TensorSpec(shape=[None, max_length], dtype=tf.int32),
                    'attention_mask': tf.TensorSpec(shape=[None, max_length], dtype=tf.int32)
                }])
                def saved_model_fn(inputs):
                    outputs = model(inputs)
                    return tf.nn.softmax(outputs.logits)
                
                # 保存 SavedModel
                tf.saved_model.save(
                    tf.Module(),
                    saved_model_path,
                    signatures={'serving_default': saved_model_fn.get_concrete_function()}
                )
                
                # 从 SavedModel 转换为 TFLite
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                tflite_success = True
                print(f"   ✅ 使用方法 2 成功转换")
            except Exception as e2:
                print(f"   ⚠️  方法 2 也失败: {str(e2)}")
                print(f"   提示: TFLite 转换遇到问题，将保存为 SavedModel 格式")
                print(f"   您可以使用 tensorflowjs_converter 手动转换为 TensorFlow.js 格式")
                
                # 保存为 SavedModel 作为备选
                saved_model_path = output_path.replace('.tflite', '_savedmodel')
                
                # 创建包装函数用于 SavedModel
                class SavedModelWrapper(tf.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    @tf.function(input_signature=[{
                        'input_ids': tf.TensorSpec(shape=[None, max_length], dtype=tf.int32),
                        'attention_mask': tf.TensorSpec(shape=[None, max_length], dtype=tf.int32)
                    }])
                    def __call__(self, inputs):
                        outputs = self.model(inputs)
                        return tf.nn.softmax(outputs.logits)
                
                wrapper = SavedModelWrapper(model)
                tf.saved_model.save(wrapper, saved_model_path)
                
                print(f"   ✅ 已保存为 SavedModel: {saved_model_path}")
                print(f"\n   转换为 TensorFlow.js 的命令:")
                print(f"   pip install tensorflowjs")
                print(f"   tensorflowjs_converter \\")
                print(f"       --input_format=tf_saved_model \\")
                print(f"       --output_format=tfjs_graph_model \\")
                print(f"       {saved_model_path} \\")
                print(f"       {saved_model_path}_js/")
                print(f"\n   或者直接使用 SavedModel 进行推理（Python 环境）")
        
        # 如果 TFLite 转换成功，保存文件
        if tflite_success and tflite_model:
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # 检查文件大小
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   ✅ TFLite 模型导出成功！")
            print(f"   模型大小: {file_size:.2f} MB")
            
            if file_size > 100:
                print(f"   ⚠️  警告: 模型文件较大（{file_size:.2f} MB），建议使用量化版本")
        else:
            print(f"   ⚠️  TFLite 模型未导出，但已保存 SavedModel 格式")
        
        # 无论 TFLite 是否成功，都保存 SavedModel（用于转换为 TensorFlow.js）
        print(f"\n💾 保存 SavedModel（用于 TensorFlow.js 转换）...")
        saved_model_path = output_path.replace('.tflite', '_savedmodel')
        
        # 创建包装函数用于 SavedModel
        class SavedModelWrapper(tf.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            @tf.function(input_signature=[{
                'input_ids': tf.TensorSpec(shape=[None, max_length], dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=[None, max_length], dtype=tf.int32)
            }])
            def __call__(self, inputs):
                outputs = self.model(inputs)
                return tf.nn.softmax(outputs.logits)
        
        wrapper = SavedModelWrapper(model)
        tf.saved_model.save(wrapper, saved_model_path)
        print(f"   ✅ SavedModel 已保存: {saved_model_path}")
        print(f"\n   转换为 TensorFlow.js 的命令:")
        print(f"   pip install tensorflowjs")
        print(f"   tensorflowjs_converter \\")
        print(f"       --input_format=tf_saved_model \\")
        print(f"       --output_format=tfjs_graph_model \\")
        print(f"       {saved_model_path} \\")
        print(f"       {saved_model_path.replace('_savedmodel', '_js')}/")
        
        # 步骤 10: 保存词汇表和标签文件
        print(f"\n💾 保存辅助文件...")
        
        # 保存词汇表
        vocab_path = output_path.replace('.tflite', '_vocab.txt')
        tokenizer.save_vocabulary(os.path.dirname(vocab_path))
        vocab_file = os.path.join(os.path.dirname(vocab_path), 'vocab.txt')
        if os.path.exists(vocab_file):
            os.rename(vocab_file, vocab_path)
        print(f"   ✅ 词汇表已保存: {vocab_path}")
        
        # 保存标签文件
        labels_path = output_path.replace('.tflite', '_labels.txt')
        with open(labels_path, 'w', encoding='utf-8') as f:
            for i in range(num_labels):
                f.write(f"{id_to_label[i]}\n")
        print(f"   ✅ 标签文件已保存: {labels_path}")
        
        # 保存标签映射
        label_map_path = output_path.replace('.tflite', '_label_map.json')
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump({
                'label_to_id': label_to_id,
                'id_to_label': id_to_label
            }, f, ensure_ascii=False, indent=2)
        print(f"   ✅ 标签映射已保存: {label_map_path}")
        
        # 保存训练日志
        log_path = output_path.replace('.tflite', '_training_log.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("训练日志\n")
            f.write("=" * 70 + "\n")
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"数据集: {dataset_path}\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"训练参数:\n")
            f.write(f"  训练轮数: {epochs}\n")
            f.write(f"  批次大小: {batch_size}\n")
            f.write(f"  学习率: {learning_rate}\n")
            f.write(f"  最大长度: {max_length}\n")
            f.write(f"  验证集比例: {validation_split}\n\n")
            f.write(f"数据统计:\n")
            f.write(f"  训练集: {len(train_df)} 条\n")
            f.write(f"  验证集: {len(val_df)} 条\n\n")
            f.write(f"评估结果:\n")
            f.write(f"  验证集准确率: {accuracy:.4f}\n\n")
            f.write("分类报告:\n")
            f.write(report)
        
        print(f"   💾 训练日志已保存: {log_path}")
        
        print("\n" + "=" * 70)
        print("✅ 训练完成！")
        print("=" * 70)
        print(f"\n输出文件:")
        print(f"  - 模型文件: {output_path}")
        print(f"  - 词汇表: {vocab_path}")
        print(f"  - 标签文件: {labels_path}")
        print(f"  - 标签映射: {label_map_path}")
        print(f"  - 评估报告: {report_path}")
        print(f"  - 训练日志: {log_path}")
        print(f"\n下一步:")
        print(f"  1. 将模型文件部署到 Web 应用")
        print(f"  2. 更新 src/config.ts 中的模型路径")
        print(f"  3. 使用 TensorFlow.js 加载模型进行推理")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误:")
        print(f"   {str(e)}")
        print(f"\n可能的解决方案:")
        print(f"  1. 检查数据集格式是否正确")
        print(f"  2. 确保数据量足够（建议至少 1000 条）")
        print(f"  3. 检查内存是否充足（BERT 模型需要较大内存）")
        print(f"  4. 尝试减小 batch_size 或 max_length")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='使用 TensorFlow 和 Transformers 训练中文情感分析 BERT 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本训练
  python3 train_bert_tensorflow.py --dataset dataset.csv

  # 自定义参数
  python3 train_bert_tensorflow.py \\
      --dataset dataset.csv \\
      --output models/my_model.tflite \\
      --epochs 5 \\
      --batch-size 8 \\
      --max-length 256
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='数据集 CSV 文件路径（必须包含 text 和 label 列）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/chinese_bert_model.tflite',
        help='输出模型文件路径（默认: models/chinese_bert_model.tflite）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='训练轮数（默认: 3）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        dest='batch_size',
        help='批次大小（默认: 16，根据内存调整）'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        dest='learning_rate',
        help='学习率（默认: 2e-5）'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        dest='validation_split',
        help='验证集比例（默认: 0.2）'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        dest='max_length',
        help='最大序列长度（默认: 128，可设置为 256）'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='bert-base-chinese',
        dest='model_name',
        help='预训练模型名称（默认: bert-base-chinese）'
    )
    
    args = parser.parse_args()
    
    success = train_model(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        max_length=args.max_length,
        model_name=args.model_name
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

