#!/usr/bin/env python3
"""
导出 BERT Tokenizer 词汇表

使用方法：
    python3 export_tokenizer.py --model-name bert-base-chinese --output vocab.txt
"""

import argparse
import os
import sys

try:
    from transformers import BertTokenizer
except ImportError:
    print("❌ 错误: 未安装 transformers")
    print("   请运行: pip install transformers")
    sys.exit(1)


def export_vocab(model_name='bert-base-chinese', output_path='vocab.txt'):
    """导出 BERT tokenizer 的词汇表"""
    
    print(f"📥 加载 Tokenizer: {model_name}")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print(f"   ✅ Tokenizer 加载成功")
    except Exception as e:
        print(f"   ❌ 加载失败: {str(e)}")
        return False
    
    print(f"\n💾 导出词汇表到: {output_path}")
    
    # 保存词汇表
    tokenizer.save_vocabulary(os.path.dirname(output_path) or '.')
    
    # 如果输出路径不是默认位置，移动文件
    vocab_file = os.path.join(os.path.dirname(output_path) or '.', 'vocab.txt')
    if vocab_file != output_path and os.path.exists(vocab_file):
        import shutil
        shutil.move(vocab_file, output_path)
    
    # 检查文件
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / 1024
        with open(output_path, 'r', encoding='utf-8') as f:
            vocab_size = len(f.readlines())
        
        print(f"   ✅ 词汇表导出成功")
        print(f"   文件大小: {file_size:.2f} KB")
        print(f"   词汇数量: {vocab_size:,}")
        return True
    else:
        print(f"   ❌ 导出失败: 文件不存在")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='导出 BERT Tokenizer 词汇表',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='bert-base-chinese',
        help='预训练模型名称（默认: bert-base-chinese）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='vocab.txt',
        help='输出文件路径（默认: vocab.txt）'
    )
    
    args = parser.parse_args()
    
    success = export_vocab(args.model_name, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

