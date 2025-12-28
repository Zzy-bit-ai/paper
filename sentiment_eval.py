# -*- coding: utf-8 -*-
"""
KATE 论文情感分类专用版 - 纯本地 Ollama（qwen:0.5b / 1.8b）
专为 Table 1 设计：SST-2_train.tsv → IMDB_dev.tsv
评估标准：Accuracy (GLUE)
"""

import os
import pandas as pd
import numpy as np
import random
import pickle
import argparse
import re
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ==================== 参数配置 ====================
parser = argparse.ArgumentParser(description="KATE 情感分类 - Accuracy 评估")
parser.add_argument('--train_file', type=str, default='SST-2_train.tsv', help='训练集文件名')
parser.add_argument('--dev_file', type=str, default='IMDB_dev.tsv', help='验证集文件名')
parser.add_argument('--epochs', type=int, default=5, help='重复运行次数')
parser.add_argument('--k', type=int, default=8, help='in-context 示例数量')
parser.add_argument('--sample_size', type=int, default=2000, help='每轮评估样本数')
parser.add_argument('--mode', type=str, default='random', choices=['random', 'kate'], help='random=随机，kate=kNN')
parser.add_argument('--kNN_dev_train', type=str, default=None, help='kNN 文件路径 (kate 模式)')
parser.add_argument('--ollama_model', type=str, default='qwen:1.8b', choices=['qwen:0.5b', 'qwen:1.8b'])
parser.add_argument('--max_workers', type=int, default=4)  # 限制并发数为 4，防止 Ollama 超时
args = parser.parse_args()

# ==================== 数据预处理 ====================
def safe_normalize(df):
    # 自动识别标签列并统一格式
    label_col = 'label' if 'label' in df.columns else 'Sentiment'
    df[label_col] = df[label_col].apply(lambda x: 1 if str(x).lower() in ['1', 'positive', 'pos'] else 0)
    # 统一文本列名
    for col in ['sentence', 'Sentence', 'text', 'Text']:
        if col in df.columns:
            df = df.rename(columns={col: 'text'})
            break
    return df

train_df = pd.read_csv(f"dataset/{args.train_file}", sep='\t', header=0, on_bad_lines='skip')
dev_df = pd.read_csv(f"dataset/{args.dev_file}", sep='\t', header=0, on_bad_lines='skip')

# 规范化标签和列名
train_df = safe_normalize(train_df)
dev_df = safe_normalize(dev_df)

print(f"训练集: {len(train_df)} 条 | 验证集: {len(dev_df)} 条")

# ==================== Ollama 生成函数 ====================
def ollama_generate(prompt: str):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": args.ollama_model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
        "stop": ["\n", "Q:", "Review:"],
        "options": {"num_predict": 10}
    }
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        print(f"[Ollama Error] {e}")
        return ""

# ==================== 高并发生成器 ====================
class LocalGenerator:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=args.max_workers)
        print(f"[本地模型] {args.ollama_model} | 并发: {args.max_workers}")

    def __call__(self, prompts):
        # 使用 map 保持顺序
        results = list(tqdm(self.executor.map(ollama_generate, prompts), 
                            total=len(prompts), desc="生成中", leave=False))
        return results

    def __del__(self):
        self.executor.shutdown(wait=True)

gen = LocalGenerator()

# ==================== 准确率计算 ====================
def classification_accuracy(pred: str, ref: str) -> bool:
    pred = pred.lower().strip()
    # 优先检查开头，减少 "not positive" 类的干扰
    if pred.startswith(('pos', '1', 'good', 'great')):
        pred_label = 'positive'
    elif pred.startswith(('neg', '0', 'bad', 'terrible')):
        pred_label = 'negative'
    else:
        # 备用：关键词包含逻辑
        if 'positive' in pred: pred_label = 'positive'
        elif 'negative' in pred: pred_label = 'negative'
        else: pred_label = 'unknown'
    
    return pred_label == ref

# ==================== 设置随机种子 ====================
random.seed(42)
np.random.seed(42)

# ==================== 加载 kNN (kate 模式) ====================
knn_indices = None
# 修改 kNN 加载逻辑
if args.mode == 'kate':
    if not args.kNN_dev_train:
        # 如果命令行没传，就用你提供的这个绝对路径作为默认值
        knn_path = r"D:\py_code\paper\KATEGPT3-main\KATEGPT3-main\inference\kNN_pretraining\IMDB_roberta-large_euclidean_CLS.dat"
    else:
        # 如果传了参数，则在指定目录下寻找
        knn_path = os.path.join("kNN_pretraining", f"{args.kNN_dev_train}.dat")
    
    print(f"正在加载 kNN 文件: {knn_path}")
    
    if not os.path.exists(knn_path):
        raise FileNotFoundError(f"找不到 kNN 文件，请检查路径: {knn_path}")

    with open(knn_path, "rb") as f:
        data = pickle.load(f)

    knn_indices = data.get("kNN_dev_train")
    if knn_indices is None:
        knn_indices = data.get("indices")
    if knn_indices is None:
        knn_indices = data.get("kNN_indices")

    # 3. 最终检查
    if knn_indices is None:
        print("--- 调试信息：kNN 文件内容 ---")
        print(f"文件包含的键: {list(data.keys())}")
        raise ValueError(f"kNN 文件中不包含任何有效的索引数据。当前文件的键为: {list(data.keys())}")

    # 确保是 numpy 数组
    knn_indices = np.array(knn_indices)
    print(f"kNN 加载成功! 数据形状: {knn_indices.shape}")


# ==================== Prompt 构建 ====================
def build_prompt(dev_idx: int):
    if args.mode == 'random':
        examples = train_df.sample(args.k)
    else:
        ids = knn_indices[dev_idx][:args.k]
        examples = train_df.iloc[ids]

    prompt = "Instruction: Classify the sentiment of the following movie reviews as positive or negative.\n\n"
    for _, row in examples.iterrows():
        sent = row.get('text', '')
        label = "positive" if row['label'] == 1 else "negative"
        prompt += f"Review: {sent}\nSentiment: {label}\n\n"
    
    test_sent = dev_df.iloc[dev_idx].get('text', '')
    prompt += f"Review: {test_sent}\nSentiment:"
    return prompt

# ==================== 主实验循环 ====================
results = []

for epoch in range(1, args.epochs + 1):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{args.epochs} | {args.ollama_model} | {args.mode.upper()} | k={args.k}")
    print(f"{'='*70}")

    total = min(args.sample_size, len(dev_df))
    indices = random.sample(range(len(dev_df)), total)

    correct = 0
    prompts = []
    refs = []

    for i in tqdm(indices, desc="构造 Prompt"):
        prompt = build_prompt(i)
        ref = "positive" if dev_df.iloc[i]['label'] == 1 else "negative"
        prompts.append(prompt)
        refs.append(ref)

    print("开始并发推理...")
    preds = gen(prompts)

    for pred, ref in zip(preds, refs):
        if classification_accuracy(pred, ref):
            correct += 1

    acc = correct / total * 100
    results.append(acc)
    print(f"Epoch {epoch} Accuracy: {acc:.2f}% ({correct}/{total})")

# ==================== 最终结果 ====================
mean = np.mean(results)
std = np.std(results)

print(f"\n{'='*70}")
print(f"最终结果")
print(f"任务: {args.dev_file} (from {args.train_file})")
print(f"模型: {args.ollama_model}")
print(f"模式: {args.mode.upper()} (k={args.k})")
print(f"平均 Accuracy: {mean:.2f}% ± {std:.2f}")
print(f"所有轮次: {', '.join(f'{x:.2f}' for x in results)}")
print(f"{'='*70}")

# ==================== 保存结果 ====================
os.makedirs("result", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = f"result/sentiment_{args.ollama_model.replace(':', '')}_{args.mode}_k{args.k}_{timestamp}.txt"
with open(result_file, "w", encoding="utf-8") as f:
    f.write(f"任务: {args.dev_file} (from {args.train_file})\n")
    f.write(f"模型: {args.ollama_model}\n")
    f.write(f"模式: {args.mode} | k={args.k}\n")
    f.write(f"平均 Accuracy: {mean:.2f}% ± {std:.2f}\n")
    f.write(f"所有轮次: {results}\n")
print(f"结果已保存: {result_file}")
