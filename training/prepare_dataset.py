#!/usr/bin/env python3
"""
数据集准备工具

功能：
1. 数据集格式转换
2. 数据清洗和预处理
3. 数据集划分（训练集/验证集）
4. 数据格式验证
"""

import pandas as pd
import argparse
import os
import sys
from sklearn.model_selection import train_test_split

def validate_dataset(df):
    """验证数据集格式"""
    errors = []
    
    # 检查必需的列
    if 'text' not in df.columns:
        errors.append("数据集缺少 'text' 列")
    if 'label' not in df.columns:
        errors.append("数据集缺少 'label' 列")
    
    if errors:
        return False, errors
    
    # 检查标签值
    valid_labels = ['积极', '消极', 'Positive', 'Negative', 'positive', 'negative']
    invalid_labels = df[~df['label'].isin(valid_labels)]['label'].unique()
    if len(invalid_labels) > 0:
        errors.append(f"发现无效标签: {invalid_labels.tolist()}")
    
    # 检查空值
    if df['text'].isna().any():
        errors.append("发现空的文本内容")
    if df['label'].isna().any():
        errors.append("发现空的标签")
    
    # 检查数据量
    if len(df) < 100:
        errors.append(f"数据量太少（{len(df)} 条），建议至少 1000 条")
    
    # 检查数据平衡性
    label_counts = df['label'].value_counts()
    if len(label_counts) < 2:
        errors.append("数据集只包含一个类别的标签")
    else:
        min_count = label_counts.min()
        max_count = label_counts.max()
        imbalance_ratio = min_count / max_count
        if imbalance_ratio < 0.5:
            errors.append(f"数据不平衡：最小类别 {min_count} 条，最大类别 {max_count} 条（比例 {imbalance_ratio:.2f}）")
    
    return len(errors) == 0, errors

def normalize_labels(df):
    """标准化标签格式"""
    label_mapping = {
        'Positive': '积极',
        'Negative': '消极',
        'positive': '积极',
        'negative': '消极',
        'POSITIVE': '积极',
        'NEGATIVE': '消极'
    }
    
    df['label'] = df['label'].map(label_mapping).fillna(df['label'])
    return df

def clean_text(text):
    """清洗文本"""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    # 移除多余空格
    text = ' '.join(text.split())
    return text

def prepare_dataset(input_file, output_file, validation_split=0.2, shuffle=True):
    """准备数据集"""
    print(f"📖 读取数据集: {input_file}")
    
    # 读取 CSV 文件
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        print("⚠️  UTF-8 编码失败，尝试其他编码...")
        df = pd.read_csv(input_file, encoding='gbk')
    
    print(f"   原始数据量: {len(df)} 条")
    
    # 标准化标签
    print("🔤 标准化标签格式...")
    df = normalize_labels(df)
    
    # 清洗文本
    print("🧹 清洗文本数据...")
    df['text'] = df['text'].apply(clean_text)
    
    # 移除空文本
    df = df[df['text'].str.len() > 0]
    print(f"   清洗后数据量: {len(df)} 条")
    
    # 验证数据集
    print("✅ 验证数据集格式...")
    is_valid, errors = validate_dataset(df)
    
    if not is_valid:
        print("❌ 数据集验证失败：")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("   ✅ 数据集格式正确")
    
    # 显示数据统计
    print("\n📊 数据统计：")
    print(f"   总数据量: {len(df)} 条")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"   {label}: {count} 条 ({count/len(df)*100:.1f}%)")
    
    # 划分数据集
    if validation_split > 0:
        print(f"\n📦 划分数据集（验证集比例: {validation_split}）...")
        train_df, val_df = train_test_split(
            df, 
            test_size=validation_split, 
            stratify=df['label'],
            random_state=42,
            shuffle=shuffle
        )
        
        train_file = output_file.replace('.csv', '_train.csv')
        val_file = output_file.replace('.csv', '_val.csv')
        
        train_df.to_csv(train_file, index=False, encoding='utf-8')
        val_df.to_csv(val_file, index=False, encoding='utf-8')
        
        print(f"   ✅ 训练集: {train_file} ({len(train_df)} 条)")
        print(f"   ✅ 验证集: {val_file} ({len(val_df)} 条)")
    
    # 保存完整数据集
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n💾 保存数据集: {output_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='准备中文情感分析数据集')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入数据集文件路径（CSV 格式）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset_prepared.csv',
        help='输出数据集文件路径（默认: dataset_prepared.csv）'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='验证集比例（默认: 0.2）'
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='不打乱数据'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 错误: 找不到输入文件: {args.input}")
        sys.exit(1)
    
    print("=" * 60)
    print("数据集准备工具")
    print("=" * 60)
    print()
    
    success = prepare_dataset(
        args.input,
        args.output,
        validation_split=args.validation_split,
        shuffle=not args.no_shuffle
    )
    
    if success:
        print("\n✅ 数据集准备完成！")
        print(f"\n下一步：运行训练脚本")
        print(f"   python3 train_chinese_sentiment.py --dataset {args.output}")
    else:
        print("\n❌ 数据集准备失败，请检查错误信息")
        sys.exit(1)

if __name__ == '__main__':
    main()

