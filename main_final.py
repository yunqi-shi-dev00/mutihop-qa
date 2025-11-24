"""
半导体QA生成系统 - 最终版主程序（优化版）
完全按用户要求：每次SELECT用所有子QA生成多跳问题
新增功能：
1. 支持语义embedding模式（--use-embedding）
2. 桥联合理性检查（--enable-bridge-check）
3. 全局JSON容错

使用示例:
    # 基础运行（关键词匹配）
    python main_final_new.py \
        --input /path/to/QA.jsonl \
        --output ./generated_qa \
        --model_path /path/to/model \
        --batch_size 4 \
        --target_count 100 \
        --debug
    
    # 启用语义embedding（更准确）
    python main_final_new.py \
        --input /path/to/QA.jsonl \
        --output ./generated_qa \
        --model_path /path/to/model \
        --use-embedding \
        --debug
"""

import argparse
import asyncio
import sys

from knowledge_base_new import EnhancedSemiconductorKB
from llm_client import LLMAPIClient
from agent_final_new import FinalSemiconductorQAAgent
from utils import (
    load_qa_data,
    validate_qa_data,
    generate_batch_with_monitoring,
    merge_generated_qa,
    print_usage_distribution,
    filter_by_quality  # ⭐ 新增：质量筛选功能
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='半导体QA生成系统 - 最终优化版（全面优化）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 输入输出
    parser.add_argument('--input', type=str, required=True,
                        help='输入QA数据文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录路径')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='LLM模型路径')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Tokenizer路径')
    parser.add_argument('--server_type', type=str, default='vllm',
                        choices=['vllm', 'sglang'],
                        help='推理服务器类型')
    parser.add_argument('--host', type=str, default='localhost',
                        help='服务器主机')
    parser.add_argument('--port', type=int, default=8000,
                        help='服务器端口')
    
    # 生成参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='并发批次大小')
    parser.add_argument('--target_count', type=int, default=50,
                        help='目标生成数量')
    parser.add_argument('--max_turns', type=int, default=16,
                        help='最大迭代轮数')
    parser.add_argument('--max_hops', type=int, default=3,
                        help='最多组合的问题数量（默认3）')
    
    # 🚀 新增：embedding相关
    parser.add_argument('--use-embedding', action='store_true',
                        help='使用语义embedding查找相关QA（需要安装sentence-transformers）')
    # ========================================
    # 🔧 修复Bug 7：Embedding模型内存不足（命令行参数）
    # 修复时间：2025-11-19
    # 说明：新增--embedding-batch-size参数，允许用户根据GPU显存调整batch_size
    # 使用：--embedding-batch-size 4（默认，4GB显存）
    #       --embedding-batch-size 2（3-4GB显存）
    #       --embedding-batch-size 1（2.5-3GB显存）
    # ========================================
    parser.add_argument('--embedding-batch-size', type=int, default=4,
                        help='Embedding生成的批量大小（默认4，减少内存占用）')
    # ========================================
    # 🔧 优化10：支持自定义embedding模型路径
    # 问题：用户可能用错模型（如7B模型），导致速度慢
    # 解决：新增--embedding-model-path参数
    # 推荐：使用Qwen3-Embedding-0.6B（快速）而不是Qwen2.5-7B（慢11倍）
    # ========================================
    parser.add_argument('--embedding-model-path', type=str, default=None,
                        help='Embedding模型路径（可选，默认使用Qwen3-Embedding-0.6B）')
    # ========================================
    
    # 功能开关
    parser.add_argument('--enable_dynamic_planning', action='store_true',
                        help='启用动态规划')
    parser.add_argument('--disable_dynamic_planning', dest='enable_dynamic_planning',
                        action='store_false')
    parser.set_defaults(enable_dynamic_planning=True)
    
    parser.add_argument('--enable_qa_filtering', action='store_true',
                        help='启用问题筛选')
    parser.add_argument('--disable_qa_filtering', dest='enable_qa_filtering',
                        action='store_false')
    parser.set_defaults(enable_qa_filtering=True)
    
    parser.add_argument('--enable_answer_regeneration', action='store_true',
                        help='启用答案重生成')
    parser.add_argument('--disable_answer_regeneration', dest='enable_answer_regeneration',
                        action='store_false')
    parser.set_defaults(enable_answer_regeneration=True)
    
    # 🆕 新增：桥联检查开关
    parser.add_argument('--enable_bridge_check', action='store_true',
                        help='启用桥联合理性检查')
    parser.add_argument('--disable_bridge_check', dest='enable_bridge_check',
                        action='store_false')
    parser.set_defaults(enable_bridge_check=True)
    
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    parser.add_argument('--merge_output', action='store_true',
                        help='合并输出')
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()
    
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    
    print(f"\n{'='*80}")
    print(f"半导体QA生成系统 - 最终优化版 🚀")
    print(f"{'='*80}")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"模型: {args.model_path}")
    print(f"服务器: {args.host}:{args.port}")
    print(f"目标数量: {args.target_count}")
    print(f"\n核心特点：")
    print(f"  1. 每次SELECT用所有子QA生成多跳问题（完全按用户模板）")
    print(f"  2. 问题筛选（6大标准）")
    print(f"  3. 答案重生成（强调围绕子QA，不发散）")
    print(f"  4. 保持原版action机制不变")
    print(f"\n✨ 新增优化：")
    print(f"  • 全局JSON解析容错（3层容错机制）")
    print(f"  • 桥联合理性检查（过滤不合理组合）")
    if args.use_embedding:
        print(f"  • 语义embedding模式（更准确的相关QA查找）")
    print(f"\n功能配置：")
    print(f"  动态规划: {'✓' if args.enable_dynamic_planning else '✗'}")
    print(f"  问题筛选: {'✓' if args.enable_qa_filtering else '✗'}")
    print(f"  答案重生成: {'✓' if args.enable_answer_regeneration else '✗'}")
    print(f"  桥联检查: {'✓' if args.enable_bridge_check else '✗'}")
    print(f"  语义embedding: {'✓' if args.use_embedding else '✗'}")
    print(f"  调试模式: {'✓' if args.debug else '✗'}")
    print(f"{'='*80}\n")
    
    # 加载数据
    try:
        qa_data = load_qa_data(args.input)
        qa_data = validate_qa_data(qa_data)
        
        if len(qa_data) == 0:
            print("[ERROR] 无有效数据")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 数据加载失败: {e}")
        sys.exit(1)
    
    # 初始化知识库（支持embedding）
    try:
        # ========================================
        # 🔧 修复Bug 7：Embedding模型内存不足（参数传递）
        # 🔧 优化10：支持自定义embedding模型路径（参数传递）
        # 修复时间：2025-11-19
        # 说明：将命令行参数embedding_batch_size和embedding_model_path传递给KB
        # 效果：KB会使用用户指定的batch_size和模型路径生成embedding
        # ========================================
        kb = EnhancedSemiconductorKB(
            qa_data, 
            use_embedding=args.use_embedding,
            embedding_batch_size=args.embedding_batch_size,  # ⭐ 传递batch_size
            embedding_model_path=args.embedding_model_path  # ⭐ 传递模型路径（优化10）
        )
        # ========================================
    except Exception as e:
        print(f"[ERROR] 知识库初始化失败: {e}")
        sys.exit(1)
    
    # 初始化LLM
    try:
        llm_client = LLMAPIClient(
            model_path=args.model_path,
            server_type=args.server_type,
            host=args.host,
            port=args.port
        )
        
        if not llm_client.is_connected:
            print(f"[WARNING] LLM服务器未连接")
    except Exception as e:
        print(f"[ERROR] LLM初始化失败: {e}")
        sys.exit(1)
    
    # 初始化Agent（新增桥联检查参数）
    try:
        async with llm_client:
            agent = FinalSemiconductorQAAgent(
                knowledge_base=kb,
                llm_client=llm_client,
                tokenizer_path=args.tokenizer_path,
                max_turns=args.max_turns,
                max_hops=args.max_hops,
                use_dynamic_planning=args.enable_dynamic_planning,
                enable_qa_filtering=args.enable_qa_filtering,
                enable_answer_regeneration=args.enable_answer_regeneration,
                enable_bridge_check=args.enable_bridge_check,
                debug_mode=args.debug
            )
            
            # 批量生成
            print(f"\n{'='*80}")
            print(f"开始生成QA")
            print(f"{'='*80}\n")
            
            stats = await generate_batch_with_monitoring(
                agent=agent,
                save_path=args.output,
                batch_size=args.batch_size,
                target_count=args.target_count
            )
            
            # 知识库统计
            kb_stats = kb.get_usage_stats()
            print_usage_distribution(kb_stats)
            
            # ⭐⭐⭐ 质量筛选（High Quality Filter）⭐⭐⭐
            print(f"\n{'='*80}")
            print(f"执行质量筛选...")
            print(f"{'='*80}")
            high_quality_files = filter_by_quality(args.output)
            
            # 保存高质量QA列表
            if high_quality_files:
                high_quality_list_path = f"{args.output}/high_quality_qa_list.json"
                import json
                with open(high_quality_list_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'total_generated': stats['successful'],
                        'high_quality_count': len(high_quality_files),
                        'quality_rate': len(high_quality_files) / stats['successful'] if stats['successful'] > 0 else 0,
                        'high_quality_files': [f.split('/')[-1] for f in high_quality_files]
                    }, f, ensure_ascii=False, indent=2)
                print(f"[SAVE] 高质量QA列表已保存: {high_quality_list_path}")
            
            # 合并输出
            if args.merge_output:
                merge_generated_qa(args.output)
            
            print(f"\n{'='*80}")
            print(f"生成完成！ 🎉")
            print(f"{'='*80}")
            print(f"总生成数: {stats['successful']}")
            print(f"高质量QA数: {len(high_quality_files)}")
            print(f"质量通过率: {len(high_quality_files) / stats['successful'] * 100:.1f}%" if stats['successful'] > 0 else "N/A")
            print(f"平均轮数: {stats.get('avg_turns', 0):.2f}")
            print(f"输出目录: {args.output}")
            
            # embedding统计
            if args.use_embedding and kb.use_embedding:
                print(f"\n💡 语义embedding模式已启用")
                print(f"   - 模型: all-MiniLM-L6-v2")
                print(f"   - QA数量: {len(kb.qa_embeddings)}")
                print(f"   - 向量维度: {kb.qa_embeddings.shape[1]}")
            
            print(f"{'='*80}\n")
            
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 用户中断")
        print(f"[INFO] 已保存文件在: {args.output}")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
