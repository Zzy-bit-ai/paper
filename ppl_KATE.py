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
from io import StringIO
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import requests
import json
from utils import categories, templates, chunks, constructPrompt, cleanLabel, most_common
from dotenv import load_dotenv

# ==================== 新增：OpenAI 兼容客户端 ====================
from openai import OpenAI

# 加载环境变量
load_dotenv()

# Ollama 配置
OLLMA_API_URL = os.getenv("OLLMA_API_URL", "http://localhost:11434/api")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLMA_MODEL", "qwen2:7b")

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ==================== 参数解析 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--task_type', default='classification', type=str,
                    help="classification or generation")
parser.add_argument('--task_name', default='SST-2', type=str)
parser.add_argument('--train_name', default='', type=str)
parser.add_argument('--category_name', default='', type=str)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--bz_train', default=1, type=int)
parser.add_argument('--knn_num', default=1, type=int)
parser.add_argument('--bz_dev', default=1, type=int)
parser.add_argument('--max_tokens', default=10, type=int)
parser.add_argument('--digits', action='store_true')
parser.add_argument('--truncate', action='store_true')
parser.add_argument('--kNN_dev_train', default='', type=str)
parser.add_argument('--evaluate_train', action='store_true')
parser.add_argument('--evaluate_train_indices', nargs='+', default=[-1], type=int)
parser.add_argument('--evaluate_dev', action='store_true')
parser.add_argument('--PIK_name', default="tmp", type=str)

# 新增：后端选择
parser.add_argument('--backend', type=str, default='ollama', choices=['ollama', 'deepseek'],required=True,
                    help="选择推理后端: ollama (本地) 或 deepseek (在线API)")

# 模型参数
parser.add_argument('--ollama_model', type=str, default=DEFAULT_OLLAMA_MODEL,
                    help="Ollama 本地模型名称 (e.g., qwen2:7b, llama3:8b)")
parser.add_argument('--deepseek_model', type=str, default=DEFAULT_DEEPSEEK_MODEL,
                    help="DeepSeek 模型名称 (e.g., deepseek-chat, deepseek-coder)")

args = parser.parse_args()

# ==================== Ollama 单次调用函数 ====================
def call_ollama_single(prompt, model=None, temperature=0.0, stop=["\n"]):
    url = f"{OLLMA_API_URL}/generate"
    payload = {
        "model": model or args.ollama_model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
        "stop": stop
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[Ollama] 调用失败: {e}")
        return ""

# ==================== 统一的 glueComplete 类（支持双后端） ====================
class glueComplete:
    def __init__(self, max_workers=12):
        self.backend = args.backend
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        if self.backend == "deepseek":
            if not DEEPSEEK_API_KEY:
                raise ValueError("使用 deepseek 后端必须在 .env 中设置 DEEPSEEK_API_KEY")
            self.client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,  # 标准 URL，无需 /beta
                timeout=60,
                max_retries=2
            )
            self.model = args.deepseek_model
            print(f"[DeepSeek] 已启用在线API，模型: {self.model} (使用 Chat Completions)")
        else:
            self.model = args.ollama_model
            print(f"[Ollama] 已启用本地模型: {self.model}")

    def __call__(self, example="", max_tokens=5):
        if isinstance(example, list):
            return self._batch_completion(example, max_tokens)
        else:
            return [self._single_completion(example, max_tokens)]

    def _batch_completion(self, prompts, max_tokens):
        if not prompts:
            return []
        futures = [self.executor.submit(self._single_completion, p, max_tokens) for p in prompts]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
        return results

    def _single_completion(self, prompt, max_tokens):
        if self.backend == "deepseek":
            return self._deepseek_completion(prompt, max_tokens)
        else:
            return self._ollama_completion(prompt, max_tokens)

    def _deepseek_completion(self, prompt, max_tokens):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions concisely based on provided examples."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                stop=["\n", "<|EOT|>"],
                top_p=1.0,
            )
            text = response.choices[0].message.content.strip()
            return text
        except Exception as e:
            print(f"[DeepSeek] 调用异常: {e}")
            return ""

    def _ollama_completion(self, prompt, max_tokens):
        return call_ollama_single(
            prompt=prompt,
            model=self.model,
            temperature=0.0,
            stop=["\n"]
        )

    def __del__(self):
        self.executor.shutdown(wait=True)

# ==================== 其余代码保持不变 ====================

# 正则预编译
PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
ARTICLE_PATTERN = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r'\s+')

PREFIXES_TO_REMOVE = [
    'The answer is: ', 'the answer is: ',
    'Answer is: ', 'answer is: ',
    'Response: ', 'response: ',
    'Answer: ', 'answer: ',
    'A: ', 'B: ', 'C: ', 'D: ', 'E: ', 'a: '
]

def clean_prediction(pred):
    if not isinstance(pred, str):
        pred = str(pred)
    for prefix in PREFIXES_TO_REMOVE:
        if pred.startswith(prefix):
            pred = pred[len(prefix):]
            break
    return pred.strip()

def compute_exact_match(pred, ref):
    def standardize(text, is_prediction=False):
        if not isinstance(text, str):
            text = str(text)
        if is_prediction:
            text = clean_prediction(text)
        text = PUNCTUATION_PATTERN.sub('', text)
        text = ARTICLE_PATTERN.sub('', text)
        text = WHITESPACE_PATTERN.sub(' ', text).lower().strip()
        return text
    return 1 if standardize(pred, True) == standardize(ref) else 0

# 初始化任务（自动选择后端）
task = glueComplete(max_workers=15)

# ==================== 数据读取与实验主循环（完全保持你原有逻辑） ====================
task_name = args.task_name
train_name = args.train_name or task_name
category_name = args.category_name or task_name

train_fname = os.path.join("dataset", f"{train_name}_train.tsv")
dev_fname = os.path.join("dataset", f"{task_name}_dev.tsv")

train_df = pd.read_csv(train_fname, sep=r'(?<![\t].)\t', engine='python', header='infer', keep_default_na=False)
dev_df = pd.read_csv(dev_fname, sep=r'(?<![\t].)\t', engine='python', header='infer', keep_default_na=False)
print(f"验证集大小: {len(dev_df)}, 训练集大小: {len(train_df)}")

label_col = categories[category_name]["A"][0].lower()
train_labels = train_df[label_col].tolist()
dev_labels = dev_df[label_col].tolist()

dev_labels_cache = {idx: dev_df.loc[idx, label_col] for idx in range(len(dev_df))}

if args.task_type == "classification":
    dev_unique_labels = sorted(list(set(dev_labels)))
    dev_indices = [[] for _ in dev_unique_labels]
    for idx, label in enumerate(dev_labels):
        dev_indices[dev_unique_labels.index(label)].append(idx)
elif args.task_type == "generation":
    dev_unique_labels = [0]
    dev_indices = [list(range(len(dev_df)))]

track_train = []
track_dev = [[] for _ in dev_unique_labels]
pred_dev = [[] for _ in dev_unique_labels]
accuracy_dev = [[] for _ in dev_unique_labels]

if args.evaluate_dev:
    for i in range(len(dev_unique_labels)):
        track_dev[i].append(dev_indices[i])
else:
    random.seed(30)
    for i in range(len(dev_unique_labels)):
        sample_num = min(args.bz_dev, len(dev_indices[i]))
        track_dev[i].append(random.sample(dev_indices[i], k=sample_num))
        print(f"类别 {i} 采样验证样本数: {sample_num}")

kNN_dev_train = None
if args.kNN_dev_train:
    PIK_kNN = os.path.join("kNN_pretraining", f"{args.kNN_dev_train}.dat")
    with open(PIK_kNN, "rb") as f:
        kNN_data = pickle.load(f)
        kNN_dev_train = kNN_data["kNN_dev_train"]
    print("使用 KATE (kNN) 方法")

Q_list = categories[category_name]["Qs"]
A_list = categories[category_name]["A"]
templates = templates[category_name]

output_buffer = StringIO()

for epoch in tqdm(range(args.epochs), desc="总进度"):
    if args.evaluate_train:
        tmp_train_indices = args.evaluate_train_indices
    else:
        tmp_train_indices = random.sample(range(len(train_df)), k=args.bz_train)
    track_train.append(tmp_train_indices)

    base_example = constructPrompt(
        df=train_df, labels=train_labels, indices=tmp_train_indices,
        templates=templates, Q_list=Q_list, A_list=A_list,
        A_flag=True, truncate=args.truncate
    )

    for cat_idx in range(len(dev_unique_labels)):
        current_dev_indices = track_dev[cat_idx][0]
        pred_dev[cat_idx].append([])

        total_em = 0
        total_samples = 0

        for batch in chunks(current_dev_indices, n=20):
            prompts = []
            batch_refs = []

            for dev_idx in batch:
                batch_refs.append(dev_labels_cache[dev_idx])

                if args.kNN_dev_train:
                    knn_indices = kNN_dev_train[dev_idx][:args.knn_num]
                    example = constructPrompt(
                        df=train_df, labels=train_labels, indices=knn_indices,
                        templates=templates, Q_list=Q_list, A_list=A_list,
                        A_flag=True, truncate=args.truncate
                    )
                else:
                    example = base_example

                question = constructPrompt(
                    df=dev_df, labels=dev_labels, indices=[dev_idx],
                    templates=templates, Q_list=Q_list, A_list=A_list,
                    A_flag=False, truncate=args.truncate
                )
                prompts.append(example + question)

            batch_preds = task(example=prompts, max_tokens=args.max_tokens)
            pred_dev[cat_idx][-1].extend(batch_preds)

            assert len(batch_preds) == len(batch_refs)

            if args.task_type == "generation":
                batch_em = sum(compute_exact_match(p, r) for p, r in zip(batch_preds, batch_refs))
                total_em += batch_em
                total_samples += len(batch_refs)

        # 记录指标
        if args.task_type == "classification":
            target = dev_unique_labels[cat_idx]
            acc = sum(1 for p in pred_dev[cat_idx][-1] if p == target) / len(pred_dev[cat_idx][-1])
            accuracy_dev[cat_idx].append(acc)
        else:
            avg_em = total_em / total_samples if total_samples > 0 else 0
            accuracy_dev[cat_idx].append(avg_em)

    # 打印本 epoch 结果
    print(f"\n=== Epoch {epoch+1}/{args.epochs} 完成 ===\n")
    for cat_idx, label in enumerate(dev_unique_labels):
        preds = pred_dev[cat_idx][-1]
        if args.task_type == "classification":
            acc = accuracy_dev[cat_idx][-1]
            print(f"类别 {label}: {acc:.4f}")
        else:
            print(f"生成任务 EM: {accuracy_dev[cat_idx][-1]:.4f}")

# ==================== 保存结果 ====================
result_dir = os.path.join("result", task_name)
os.makedirs(result_dir, exist_ok=True)
PIK = os.path.join(result_dir, f"{args.PIK_name}_em_evaluation.dat")

data = {
    "task": task_name,
    "task_type": args.task_type,
    "backend": args.backend,
    "model": task.model if hasattr(task, 'model') else args.ollama_model,
    "em_scores": accuracy_dev,
    "pred_dev": pred_dev,
    "config": vars(args)
}

with open(PIK, "wb") as f:
    pickle.dump(data, f)

print(f"\n实验完成！结果已保存至: {PIK}")