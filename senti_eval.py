# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import pickle
import argparse
import re
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
import requests
import json

load_dotenv()

# ==================== 环境变量 ====================
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL  = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")   # 标准地址（chat）
OLLAMA_API_URL     = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "qwen:1.8b")
DEEPSEEK_MODEL     = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ==================== 参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='sst2')
parser.add_argument('--train_name', type=str, default='SST-2')
parser.add_argument('--dev_name', type=str, default='IMDB')
parser.add_argument('--category', type=str, default='sentiment')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--bz_dev', type=int, default=200)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--max_tokens', type=int, default=10)
parser.add_argument('--kNN_dev_train', type=str, 
                    default='SST-2_train_dev_full_roberta-large-nli-mean-tokens_cosine_mean_with_distances')
parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'kate'])
parser.add_argument('--eval_size', type=int, default=2000)

# ==================== 新增：后端选择 ====================
parser.add_argument('--backend', type=str, default='ollama', choices=['ollama', 'deepseek'],
                    help='推理后端：ollama（本地）或 deepseek（在线API）')
parser.add_argument('--ollama_model', type=str, default=OLLAMA_MODEL,
                    help='本地 Ollama 模型名称')
parser.add_argument('--deepseek_model', type=str, default=DEEPSEEK_MODEL,
                    help='DeepSeek 模型名称（默认 deepseek-chat）')

args = parser.parse_args()

# ==================== Ollama 单次调用工具函数 ====================
def ollama_generate(prompt: str, model: str = None, max_tokens: int = 10):
    url = f"{OLLAMA_API_URL}/generate"
    payload = {
        "model": model or args.ollama_model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
        "stop": ["\n"],
        "options": {"num_predict": max_tokens}
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        print(f"[Ollama] 调用失败: {e}")
        return ""

# ==================== 统一的生成器（支持双后端）===================
class Generator:
    def __init__(self, max_workers=15):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        if args.backend == "deepseek":
            if not DEEPSEEK_API_KEY:
                raise ValueError("使用 deepseek 后端必须设置 DEEPSEEK_API_KEY")
            self.client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                timeout=60,
                max_retries=3
            )
            self.model = args.deepseek_model
            print(f"[DeepSeek] 已启用，模型: {self.model}")
        else:
            self.model = args.ollama_model
            print(f"[Ollama] 已启用，模型: {self.model}")

    def __call__(self, prompts, max_tokens=10):
        if isinstance(prompts, str):
            prompts = [prompts]
        futures = [self.executor.submit(self.single, p, max_tokens) for p in prompts]
        results = []
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
        return results

    def single(self, prompt: str, max_tokens: int):
        if args.backend == "deepseek":
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    stop=["\n"],
                    top_p=1.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[DeepSeek] 生成失败: {e}")
                return ""
        else:
            return ollama_generate(prompt, model=self.model, max_tokens=max_tokens)

    def __del__(self):
        self.executor.shutdown(wait=True)

# ==================== 实例化生成器（自动选择后端）===================
gen = Generator(max_workers=20)   # 并发数可以调高，DeepSeek 扛得住

# ==================== 工具函数（保持不变）===================
def normalize(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

def normalize_similarity(dist):
    return 1 - normalize(dist)

def compute_accuracy(pred, ref):
    def clean(s):
        s = str(s).lower().strip()
        if 'positive' in s or 'pos' in s:
            return 'positive'
        elif 'negative' in s or 'neg' in s:
            return 'negative'
        return 'invalid'
    return clean(pred) == ref.lower()

# ==================== 加载数据（保持不变）===================
train_df = pd.read_csv(f"dataset/{args.train_name}_train.tsv", sep='\t', header=0)
dev_df   = pd.read_csv(f"dataset/{args.dev_name}_dev.tsv",   sep='\t', header=0)

# ==================== 加载KNN（KATE模式）===================
if args.mode == 'kate':
    knn_path = f"kNN_pretraining/{args.kNN_dev_train}.dat"
    with open(knn_path, "rb") as f:
        knn = pickle.load(f)

    indices = None
    distances = None
    for key in ["kNN_dev_train", "indices", "index", "knn_indices"]:
        if key in knn and knn[key] is not None:
            indices = np.array(knn[key], dtype=np.int64)
            break
    for key in ["kNN_distances", "distances", "distance", "knn_distances"]:
        if key in knn and knn[key] is not None:
            distances = np.array(knn[key], dtype=np.float32)
            break

    if indices is None:
        raise ValueError("KNN 文件中未找到有效的 indices")
    if distances is None:
        print("未找到距离信息，使用伪距离")
        distances = np.tile(np.arange(20) * 0.05, (len(dev_df), 1))

    print(f"KNN 加载成功 → indices: {indices.shape}, distances: {distances.shape}")

# ==================== 主实验循环（保持不变）===================
for epoch in range(args.epochs):
    test_idx = random.sample(range(len(dev_df)), min(args.eval_size, len(dev_df)))
    batches = [test_idx[i:i+args.bz_dev] for i in range(0, len(test_idx), args.bz_dev)]

    correct = 0
    total   = len(test_idx)

    for batch in tqdm(batches, desc=f"Epoch {epoch+1} 处理批次"):
        prompts = []
        refs    = []

        for idx in batch:
            sentence = dev_df.iloc[idx]['sentence']
            label    = dev_df.iloc[idx]['label']
            q_prompt = f"Review: {sentence}\nSentiment: "

            if args.mode == 'baseline':
                example_ids = random.sample(range(len(train_df)), args.k)
            else:  # kate
                example_ids = indices[idx][:args.k]

            knn_prompt = "".join(
                f"Review: {train_df.iloc[i]['sentence']}\nSentiment: {train_df.iloc[i]['label']}\n\n"
                for i in example_ids
            )

            full_prompt = knn_prompt + q_prompt
            prompts.append(full_prompt)
            refs.append(label)

        preds = gen(prompts, max_tokens=args.max_tokens)

        for pred, ref in zip(preds, refs):
            if compute_accuracy(pred, ref):
                correct += 1

    acc = correct / total if total > 0 else 0
    print(f"\nEpoch {epoch+1} Accuracy: {acc:.4f}  ({correct}/{total})\n")

print("实验完成！")