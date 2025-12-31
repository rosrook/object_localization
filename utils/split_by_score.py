#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据分数将结果文件分流到两个文件
以指定分数为界，将数据分为高分和低分两组
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


def split_by_score(
    input_file: Path,
    high_score_file: Path,
    low_score_file: Path,
    threshold: float = 0.6,
    include_equal: bool = True,
    include_no_score_in_high: bool = False
) -> Tuple[int, int]:
    """
    根据分数将结果分流到两个文件
    
    Args:
        input_file: 输入JSON文件路径
        high_score_file: 高分输出文件路径（>= threshold）
        low_score_file: 低分输出文件路径（< threshold）
        threshold: 分数阈值（默认0.6）
        include_equal: 等于阈值的记录是否归入高分组（默认True）
        
    Returns:
        (高分记录数, 低分记录数)
    """
    # 读取输入文件
    print(f"[INFO] 读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")
    
    total = len(data)
    print(f"[INFO] 总记录数: {total}")
    print(f"[INFO] 分数阈值: {threshold} (等于阈值{'归入' if include_equal else '不归入'}高分组)")
    
    # 分流
    high_score_data = []
    low_score_data = []
    no_score_data = []  # 没有分数的记录（错误记录等）
    
    for item in data:
        total_score = item.get("total_score")
        
        if total_score is None:
            # 没有分数（可能是错误记录）
            no_score_data.append(item)
        elif include_equal and total_score >= threshold:
            high_score_data.append(item)
        elif not include_equal and total_score > threshold:
            high_score_data.append(item)
        else:
            low_score_data.append(item)
    
    # 打印统计信息
    print(f"\n[统计] 分流结果:")
    print(f"  高分组 (>= {threshold}): {len(high_score_data)} 条")
    print(f"  低分组 (< {threshold}): {len(low_score_data)} 条")
    print(f"  无分数记录: {len(no_score_data)} 条")
    
    if high_score_data:
        high_scores = [item.get("total_score", 0) for item in high_score_data if "total_score" in item]
        if high_scores:
            print(f"  高分组分数范围: {min(high_scores):.3f} - {max(high_scores):.3f}")
            print(f"  高分组平均分数: {sum(high_scores) / len(high_scores):.3f}")
    
    if low_score_data:
        low_scores = [item.get("total_score", 0) for item in low_score_data if "total_score" in item]
        if low_scores:
            print(f"  低分组分数范围: {min(low_scores):.3f} - {max(low_scores):.3f}")
            print(f"  低分组平均分数: {sum(low_scores) / len(low_scores):.3f}")
    
    # 保存高分文件
    if high_score_data or no_score_data:
        # 可以选择是否将无分数记录也保存到高分文件
        high_output = high_score_data + (no_score_data if include_no_score_in_high else [])
        
        high_score_file.parent.mkdir(parents=True, exist_ok=True)
        with open(high_score_file, 'w', encoding='utf-8') as f:
            json.dump(high_output, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 高分组已保存到: {high_score_file} ({len(high_output)} 条)")
    else:
        # 创建空文件
        high_score_file.parent.mkdir(parents=True, exist_ok=True)
        with open(high_score_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(f"[INFO] 高分组为空，已创建空文件: {high_score_file}")
    
    # 保存低分文件
    if low_score_data:
        low_score_file.parent.mkdir(parents=True, exist_ok=True)
        with open(low_score_file, 'w', encoding='utf-8') as f:
            json.dump(low_score_data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 低分组已保存到: {low_score_file} ({len(low_score_data)} 条)")
    else:
        # 创建空文件
        low_score_file.parent.mkdir(parents=True, exist_ok=True)
        with open(low_score_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(f"[INFO] 低分组为空，已创建空文件: {low_score_file}")
    
    # 如果需要，单独保存无分数记录
    if no_score_data and not include_no_score_in_high:
        no_score_file = low_score_file.parent / f"{low_score_file.stem}_no_score{low_score_file.suffix}"
        with open(no_score_file, 'w', encoding='utf-8') as f:
            json.dump(no_score_data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 无分数记录已保存到: {no_score_file} ({len(no_score_data)} 条)")
    
    return len(high_score_data), len(low_score_data)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='根据分数将结果文件分流到两个文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 以0.6为界分流
  python utils/split_by_score.py results.json high_score.json low_score.json
  
  # 自定义阈值
  python utils/split_by_score.py results.json high.json low.json --threshold 0.7
  
  # 等于阈值的记录归入低分组
  python utils/split_by_score.py results.json high.json low.json --threshold 0.6 --exclude-equal
  
  # 将无分数记录也保存到高分文件
  python utils/split_by_score.py results.json high.json low.json --include-no-score
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入JSON文件路径'
    )
    parser.add_argument(
        'high_score_file',
        type=str,
        help='高分输出文件路径（>= threshold）'
    )
    parser.add_argument(
        'low_score_file',
        type=str,
        help='低分输出文件路径 (< threshold)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='分数阈值（默认: 0.6）'
    )
    parser.add_argument(
        '--exclude-equal',
        action='store_true',
        help='等于阈值的记录归入低分组（默认归入高分组）'
    )
    parser.add_argument(
        '--include-no-score',
        action='store_true',
        help='将无分数记录（错误记录等）也保存到高分文件（默认不保存）'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    high_score_file = Path(args.high_score_file)
    low_score_file = Path(args.low_score_file)
    
    if not input_file.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        return
    
    if args.threshold < 0 or args.threshold > 1:
        print(f"[ERROR] 阈值应该在0-1之间，当前值: {args.threshold}")
        return
    
    try:
        high_count, low_count = split_by_score(
            input_file=input_file,
            high_score_file=high_score_file,
            low_score_file=low_score_file,
            threshold=args.threshold,
            include_equal=not args.exclude_equal,
            include_no_score_in_high=args.include_no_score
        )
        
        print(f"\n[完成] 分流完成！高分组: {high_count} 条，低分组: {low_count} 条")
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# # 以0.6为界分流（默认）
# python utils/split_by_score.py results.json high_score.json low_score.json

# # 自定义阈值0.7
# python utils/split_by_score.py results.json high.json low.json --threshold 0.7

# # 等于阈值的记录归入低分组
# python utils/split_by_score.py results.json high.json low.json --exclude-equal

# # 将无分数记录也保存到高分文件
# python utils/split_by_score.py results.json high.json low.json --include-no-score