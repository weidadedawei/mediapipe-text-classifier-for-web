#!/usr/bin/env python3
"""
数据集格式转换工具

将 weibo_senti_100k 和 ChnSentiCorp 数据集转换为训练脚本需要的格式：
- 列名：text, label
- label 值：积极（原 1）或 消极（原 0）
"""

import pandas as pd
import argparse
import os
import sys

def convert_dataset(input_file, output_file, text_column='review', label_column='label'):
    """转换数据集格式"""
    print(f"📖 读取数据集: {input_file}")
    
    try:
        # 读取 CSV 文件
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        print("⚠️  UTF-8 编码失败，尝试其他编码...")
        try:
            df = pd.read_csv(input_file, encoding='gbk')
        except:
            df = pd.read_csv(input_file, encoding='gb18030')
    
    print(f"   原始数据量: {len(df)} 条")
    print(f"   列名: {df.columns.tolist()}")
    
    # 检查必需的列
    if text_column not in df.columns:
        print(f"❌ 错误: 找不到文本列 '{text_column}'")
        print(f"   可用列: {df.columns.tolist()}")
        return False
    
    if label_column not in df.columns:
        print(f"❌ 错误: 找不到标签列 '{label_column}'")
        print(f"   可用列: {df.columns.tolist()}")
        return False
    
    # 创建新的 DataFrame
    result_df = pd.DataFrame()
    result_df['text'] = df[text_column]
    result_df['label'] = df[label_column]
    
    # 转换标签：1 -> 积极, 0 -> 消极
    print("🔄 转换标签格式...")
    label_mapping = {
        1: '积极',
        0: '消极',
        '1': '积极',
        '0': '消极',
        1.0: '积极',
        0.0: '消极'
    }
    
    result_df['label'] = result_df['label'].map(label_mapping)
    
    # 检查是否有未映射的标签
    unmapped = result_df[result_df['label'].isna()]
    if len(unmapped) > 0:
        print(f"⚠️  警告: 发现 {len(unmapped)} 条未映射的标签")
        print(f"   未映射的值: {unmapped['label'].unique().tolist()}")
        # 移除未映射的行
        result_df = result_df[result_df['label'].notna()]
    
    # 移除空文本
    result_df = result_df[result_df['text'].notna()]
    result_df = result_df[result_df['text'].astype(str).str.strip().str.len() > 0]
    
    print(f"   转换后数据量: {len(result_df)} 条")
    
    # 显示标签分布
    print("\n📊 标签分布:")
    label_counts = result_df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"   {label}: {count} 条 ({count/len(result_df)*100:.1f}%)")
    
    # 保存转换后的数据集
    print(f"\n💾 保存转换后的数据集: {output_file}")
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"   ✅ 转换完成！")
    return True

def merge_datasets(files, output_file):
    """合并多个数据集"""
    print("=" * 70)
    print("合并数据集")
    print("=" * 70)
    print()
    
    all_dataframes = []
    
    for file in files:
        print(f"📖 读取: {file}")
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='gbk')
        
        # 确保列名正确
        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"   ⚠️  跳过: 格式不正确（需要 text 和 label 列）")
            continue
        
        print(f"   ✅ {len(df)} 条数据")
        all_dataframes.append(df)
    
    if not all_dataframes:
        print("❌ 错误: 没有有效的数据集可以合并")
        return False
    
    # 合并数据
    print(f"\n🔀 合并数据集...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 去重（基于文本内容）
    print(f"   合并前: {len(merged_df)} 条")
    merged_df = merged_df.drop_duplicates(subset=['text'], keep='first')
    print(f"   去重后: {len(merged_df)} 条")
    
    # 打乱数据
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 显示统计信息
    print("\n📊 合并后统计:")
    print(f"   总数据量: {len(merged_df)} 条")
    label_counts = merged_df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"   {label}: {count} 条 ({count/len(merged_df)*100:.1f}%)")
    
    # 保存合并后的数据集
    print(f"\n💾 保存合并后的数据集: {output_file}")
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"   ✅ 合并完成！")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='转换和合并中文情感分析数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个数据集
  python3 convert_datasets.py \\
      --input datasets/weibo_senti_100k.csv \\
      --output datasets/weibo_senti_100k_converted.csv \\
      --text-column review \\
      --label-column label

  # 转换并合并多个数据集
  python3 convert_datasets.py \\
      --merge \\
      --inputs datasets/weibo_senti_100k_converted.csv datasets/ChnSentiCorp_converted.csv \\
      --output datasets/dataset_merged.csv
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='输入数据集文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出数据集文件路径'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='review',
        help='文本列名（默认: review）'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default='label',
        help='标签列名（默认: label）'
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='合并多个数据集'
    )
    parser.add_argument(
        '--inputs',
        nargs='+',
        help='要合并的数据集文件列表（使用 --merge 时）'
    )
    
    args = parser.parse_args()
    
    if args.merge:
        # 合并模式
        if not args.inputs or not args.output:
            print("❌ 错误: 合并模式需要 --inputs 和 --output 参数")
            sys.exit(1)
        
        success = merge_datasets(args.inputs, args.output)
    else:
        # 转换模式
        if not args.input or not args.output:
            print("❌ 错误: 转换模式需要 --input 和 --output 参数")
            sys.exit(1)
        
        success = convert_dataset(
            args.input,
            args.output,
            text_column=args.text_column,
            label_column=args.label_column
        )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

