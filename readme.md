# 🚀 从这里开始

## 📦 最终文件（共13个）

### ⭐ 核心代码（7个）
```
main_final.py          - 主程序入口
agent_final.py         - 增强Agent（多跳+筛选+重生成+动态规划）
prompts_final.py       - Prompt模板（完全按用户模板）
knowledge_base.py      - 知识库（动态规划+消耗统计）
llm_client.py          - LLM客户端
utils.py               - 工具函数
requirements.txt       - 依赖包
```

### 📖 文档（5个）
```
最终_readme.md         - 📘 完整使用指南（从这里开始）
功能说明.md            - 📗 详细功能机制
功能对比总结.md        - 📙 原版vs最终版对比
检查清单.md            - 📕 修复清单和测试建议
文件清单.md            - 📓 文件用途说明
```

### 🔖 参考（1个）
```
semiconductor_qa_agent.py  - 原版代码（对比参考）
```

---

## ⚡ 快速开始

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

### 2️⃣ 运行测试
```bash
python main_final.py \
    --input /path/to/QA.jsonl \
    --output ./test_output \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --batch_size 2 \
    --target_count 10 \
    --max_hops 3 \
    --debug
```

### 3️⃣ 生产运行
```bash
python main_final.py \
    --input /path/to/QA.jsonl \
    --output ./generated_qa \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --batch_size 4 \
    --target_count 100 \
    --max_turns 16 \
    --max_hops 3 \
    --enable_dynamic_planning \
    --enable_qa_filtering \
    --enable_answer_regeneration \
    --debug
``

`
CUDA_VISIBLE_DEVICES=1

python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data/LLM/lhy/models/Qwen3/Qwen3-0.6B \
    --tensor-parallel-size 1\
    --disable-custom-all-reduce \
    --gpu-memory-utilization 0.6 \
    --max-model-len 1024 \
    --host 0.0.0.0 --port 8001

做多的token记得设计

python main_final.py     --input /mnt/workspace/LLM/ldd/多跳数据/data/QA-all.jsonl     --output ./generated_qa_all     --model_path /mnt/storage/models/Qwen/Qwen/Qwen3-235B-A22B-Instruct-2507     --tokenizer_path /mnt/storage/models/Qwen/Qwen/Qwen3-235B-A22B-Instruct-2507     --host localhost     --port 8000     --batch_size 32     --target_count 3000    --max_turns 12     --max_hops 4     --use-embedding     --enable_dynamic_planning     --enable_qa_filtering     --enable_answer_regeneration     --enable_bridge_check     --debug     --merge_output --embedding-model-path /mnt/data/LLM/lhy/models/Qwen3/Qwen3-4B


merge_qa.jsonl为所有得QA
/mnt/data/LLM/lhy/models/Qwen/Qwen2.5-7B-Instruct
/mnt/storage/models/Qwen/Qwen/Qwen3-235B-A22B-Instruct-2507
## 🎯 核心功能

| 功能 | 说明 | 默认状态 |
|------|------|---------|
| **多跳问题生成** | 每次SELECT用所有子QA生成N跳问题 | ✅ 启用 |
| **最大跳数限制** | `--max_hops 3` 最多组合3个问题 | ✅ 3个 |
| **动态规划策略** | 3阶段自适应选择策略 | ✅ 启用 |
| **消耗统计追踪** | 论文级别使用率统计 | ✅ 启用 |
| **问题筛选** | 6大标准评估 | ✅ 启用 |
| **答案重生成** | 围绕子QA，不发散 | ✅ 启用 |
| **原版Action机制** | SELECT/FUZZ/EXIT/BRAINSTORM | ✅ 保留 |

---

## 📚 文档导航

### 想了解如何使用？
👉 阅读 **`最终_readme.md`**

### 想了解详细功能？
👉 阅读 **`功能说明.md`**
- 最多组合几个问题？
- 动态规划如何工作？
- 消耗统计如何追踪？

### 想了解原版vs最终版区别？
👉 阅读 **`功能对比总结.md`**

### 想了解修复了哪些问题？
👉 阅读 **`检查清单.md`**

### 想了解文件用途？
👉 阅读 **`文件清单.md`**

---

## ✅ 已删除的文件

以下旧版本和中间文件已全部删除：
- ❌ `agent.py` - 旧版Agent
- ❌ `main.py` - 旧版主程序
- ❌ `prompts.py` - 旧版模板
- ❌ `README_NEW.md` - 中间文档
- ❌ `README_OPTIMIZED.md` - 中间文档
- ❌ `USAGE_GUIDE.md` - 中间文档
- ❌ `readme.md` - 原始文档
- ❌ `README.md` - 原始文档

---

## 🎉 准备就绪！

**所有文件已整理完毕，可以直接使用！**

1. ✅ 核心代码完整（7个文件）
2. ✅ 文档齐全（5个文件）
3. ✅ 原版参考保留（1个文件）
4. ✅ 所有旧版本已删除
5. ✅ 所有功能已实现
6. ✅ 所有漏洞已修复

**从 `最终_readme.md` 开始！** 📘
