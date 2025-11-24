"""
半导体QA生成系统 - 工具函数模块
包含数据加载、批处理、监控等辅助功能
"""

import json
import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from tqdm.asyncio import tqdm_asyncio


def load_qa_data(input_path: str) -> List[Dict]:
    """
    加载QA数据
    
    Args:
        input_path: 输入文件路径（支持.json和.jsonl）
    
    Returns:
        QA数据列表
    """
    print(f"\n[DATA] 加载数据: {input_path}")
    
    qa_data = []
    file_ext = Path(input_path).suffix
    
    if file_ext == '.json':
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                qa_data = data
            else:
                qa_data = [data]
    
    elif file_ext == '.jsonl':
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    qa_data.append(json.loads(line))
    
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    # 确保每个QA有ID
    for i, qa in enumerate(qa_data):
        if 'id' not in qa:
            qa['id'] = f"qa_{i:06d}"
    
    print(f"[DATA] 加载完成: {len(qa_data)} 条QA数据")
    return qa_data


def save_generation_report(output_dir: str, stats: Dict):
    """
    保存生成报告
    
    Args:
        output_dir: 输出目录
        stats: 统计信息
    """
    report_path = os.path.join(output_dir, 'generation_report.json')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n[REPORT] 生成报告已保存: {report_path}")
    
    # 打印报告摘要
    print(f"\n{'='*60}")
    print(f"生成报告摘要")
    print(f"{'='*60}")
    print(f"总生成数量: {stats.get('total_generated', 0)}")
    print(f"成功数量: {stats.get('successful', 0)}")
    print(f"失败数量: {stats.get('failed', 0)}")
    print(f"多跳QA数量: {stats.get('multihop_count', 0)}")
    print(f"通过筛选数量: {stats.get('passed_filtering', 0)}")
    print(f"答案重生成数量: {stats.get('answer_regenerated', 0)}")
    print(f"平均轮数: {stats.get('avg_turns', 0):.2f}")
    print(f"总耗时: {stats.get('total_time', 0):.2f} 秒")
    print(f"{'='*60}\n")


async def generate_batch_with_monitoring(agent, save_path: str, 
                                         batch_size: int = 4,
                                         target_count: int = 50,
                                         progress_callback: Optional[callable] = None):
    """
    批量生成QA并监控进度
    
    Args:
        agent: Agent实例
        save_path: 保存路径
        batch_size: 批次大小
        target_count: 目标数量
        progress_callback: 进度回调函数
    
    Returns:
        生成结果统计
    """
    print(f"\n{'='*80}")
    print(f"开始批量生成")
    print(f"目标数量: {target_count}")
    print(f"批次大小: {batch_size}")
    print(f"输出目录: {save_path}")
    print(f"{'='*80}\n")
    
    os.makedirs(save_path, exist_ok=True)
    
    semaphore = asyncio.Semaphore(batch_size)
    
    stats = {
        'total_generated': 0,
        'successful': 0,
        'failed': 0,
        'multihop_count': 0,
        'passed_filtering': 0,
        'answer_regenerated': 0,
        'total_turns': 0,
        'start_time': time.time()
    }
    
    async def generate_one():
        try:
            result = await agent.generate(semaphore, save_path)
            if result:
                stats['successful'] += 1
                stats['total_turns'] += result.get('num_turns', 0)
                if result.get('has_multihop'):
                    stats['multihop_count'] += 1
                if result.get('passed_filtering'):
                    stats['passed_filtering'] += 1
                if result.get('answer_regenerated'):
                    stats['answer_regenerated'] += 1
            else:
                stats['failed'] += 1
            
            stats['total_generated'] += 1
            
            # 调用进度回调
            if progress_callback:
                progress_callback(stats, target_count)
            
            # 打印进度
            if stats['total_generated'] % 10 == 0:
                print(f"\n[PROGRESS] 已完成: {stats['total_generated']}/{target_count}")
                print(f"           成功率: {stats['successful']/stats['total_generated']*100:.1f}%")
                print(f"           多跳率: {stats['multihop_count']/stats['successful']*100:.1f}%")
                
                # 打印知识库统计
                kb_stats = agent.kb.get_usage_stats()
                print(f"           论文覆盖: {kb_stats['completed_papers']}/{kb_stats['total_papers']}")
            
            return result
        except Exception as e:
            print(f"[ERROR] 生成失败: {e}")
            stats['failed'] += 1
            stats['total_generated'] += 1
            return None
    
    # 批量生成
    tasks = [generate_one() for _ in range(target_count)]
    results = await tqdm_asyncio.gather(*tasks, desc="生成QA")
    
    # 计算最终统计
    stats['end_time'] = time.time()
    stats['total_time'] = stats['end_time'] - stats['start_time']
    stats['avg_turns'] = stats['total_turns'] / stats['successful'] if stats['successful'] > 0 else 0
    
    # 添加知识库统计
    kb_stats = agent.kb.get_usage_stats()
    stats['kb_stats'] = kb_stats
    
    # 保存报告
    save_generation_report(save_path, stats)
    
    return stats


def merge_generated_qa(output_dir: str, output_file: str = 'merged_qa.jsonl'):
    """
    合并生成的QA文件
    
    Args:
        output_dir: 输出目录
        output_file: 合并后的文件名
    """
    print(f"\n[MERGE] 合并生成的QA文件...")
    
    merged_path = os.path.join(output_dir, output_file)
    merged_count = 0
    
    with open(merged_path, 'w', encoding='utf-8') as outf:
        for filename in os.listdir(output_dir):
            if filename.endswith('.json') and filename != 'generation_report.json':
                filepath = os.path.join(output_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as inf:
                        data = json.load(inf)
                        outf.write(json.dumps(data, ensure_ascii=False) + '\n')
                        merged_count += 1
                except Exception as e:
                    print(f"[WARNING] 合并文件失败 {filename}: {e}")
    
    print(f"[MERGE] 合并完成: {merged_count} 个QA文件")
    print(f"[MERGE] 输出文件: {merged_path}")
    
    return merged_path


def print_usage_distribution(kb_stats: Dict):
    """
    打印知识库使用分布
    
    Args:
        kb_stats: 知识库统计信息
    """
    print(f"\n{'='*60}")
    print(f"知识库使用分布")
    print(f"{'='*60}")
    
    usage_dist = kb_stats.get('usage_distribution', {})
    for range_name, count in usage_dist.items():
        print(f"{range_name:12s}: {count:4d} 篇论文")
    
    print(f"\n低使用率论文 (Top 10):")
    for paper, rate in kb_stats.get('low_usage_papers', []):
        print(f"  - {paper[:50]:50s} ({rate*100:5.1f}%)")
    
    print(f"{'='*60}\n")


def validate_qa_data(qa_data: List[Dict]) -> List[Dict]:
    """
    验证QA数据格式
    
    Args:
        qa_data: QA数据列表
    
    Returns:
        验证通过的QA数据列表
    """
    print(f"\n[VALIDATE] 验证QA数据格式...")
    
    required_fields = ['question', 'answer']
    valid_qa = []
    invalid_count = 0
    
    for i, qa in enumerate(qa_data):
        # 检查必需字段
        if all(field in qa for field in required_fields):
            valid_qa.append(qa)
        else:
            missing = [f for f in required_fields if f not in qa]
            print(f"[WARNING] QA {i} 缺少字段: {missing}")
            invalid_count += 1
    
    print(f"[VALIDATE] 验证完成:")
    print(f"           有效: {len(valid_qa)}")
    print(f"           无效: {invalid_count}")
    
    return valid_qa


def filter_by_quality(output_dir: str, min_confidence: float = 0.6) -> List[str]:
    """
    根据质量筛选生成的QA
    
    Args:
        output_dir: 输出目录
        min_confidence: 最小置信度
    
    Returns:
        高质量QA文件列表
    """
    print(f"\n[QUALITY] 筛选高质量QA (置信度 >= {min_confidence})...")
    
    high_quality_files = []
    
    for filename in os.listdir(output_dir):
        if filename.endswith('.json') and filename != 'generation_report.json':
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 检查质量指标
                    passed_filtering = data.get('passed_filtering', False)
                    answer_regenerated = data.get('answer_regenerated', False)
                    
                    if passed_filtering and answer_regenerated:
                        high_quality_files.append(filepath)
            except Exception as e:
                print(f"[WARNING] 读取文件失败 {filename}: {e}")
    
    print(f"[QUALITY] 找到 {len(high_quality_files)} 个高质量QA")
    
    return high_quality_files
