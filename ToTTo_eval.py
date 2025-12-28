# -*- coding: utf-8 -*-
import os
import sys
import io
import pandas as pd
import numpy as np
import random
import pickle
import argparse
import requests
import nltk
from tqdm import tqdm
import concurrent.futures
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 1. 解决 Windows 控制台乱码问题
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 2. 确保 NLTK 数据可用
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ==================== Ollama 调用封装 ====================
def call_ollama(prompt, model="qwen:1.8b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "num_predict": 128,
            "top_k": 1,
            "top_p": 0.0
        },
        "stop": ["\n", "Table:", "Description:"]
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception as e:
        return f"ERROR_CALLING_MODEL: {str(e)}"

class OllamaGenerator:
    def __init__(self, model="qwen:1.8b", workers=4):
        self.model = model
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

    def __call__(self, prompts):
        return list(self.pool.map(lambda p: call_ollama(p, self.model), prompts))

# ==================== Prompt 构建逻辑 ====================
def build_example(table, sentence):
    return f"Table: {table}\nDescription: {sentence}\n\n"

def build_prompt(context_examples, test_table):
    instruction = "Instruction: Given a Wikipedia table context, write a fluent and accurate sentence describing the highlighted cells.\n\n"
    prompt = instruction + "".join(context_examples)
    prompt += f"Table: {test_table}\nDescription:"
    return prompt

# ==================== BLEU 评估逻辑 ====================
def compute_bleu_score(preds, refs):
    smoother = SmoothingFunction().method1
    bleus = []
    for p, r in zip(preds, refs):
        if not p.strip() or "ERROR" in p:
            bleus.append(0.0)
            continue
        pred_tokens = nltk.word_tokenize(p.lower())
        ref_tokens = nltk.word_tokenize(r.lower())
        if not pred_tokens:
            bleus.append(0.0)
            continue
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
        bleus.append(score)
    return np.mean(bleus) * 100

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="ToTTo Evaluation Script (KATE/Random)")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--knn_file', type=str, default=None)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--model', type=str, default="qwen:1.8b")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--method', type=str, default='random', choices=['random', 'kate'])
    parser.add_argument('--sample_size', type=int, default=2000, help="默认评估前2000个样本")
    parser.add_argument('--epochs', type=int, default=1, help="运行轮数")
    args = parser.parse_args()

    # 1. 加载数据集
    dev_path = os.path.join(args.data_dir, "ToTTo_dev.tsv")
    train_path = os.path.join(args.data_dir, "ToTTo_train.tsv")
    
    dev_df = pd.read_csv(dev_path, sep='\t', encoding='utf-8').fillna("")
    train_df = pd.read_csv(train_path, sep='\t', encoding='utf-8').fillna("")

    if args.sample_size:
        dev_df = dev_df.head(args.sample_size)
    
    # 【修复 NameError】在此处初始化训练集索引列表
    all_train_idxs = list(range(len(train_df)))

    # 2. 准备 KATE 索引
    knn_indices = None
    if args.method == 'kate':
        with open(args.knn_file, "rb") as f:
            knn_data = pickle.load(f)
        knn_indices = knn_data.get("kNN_dev_train")
        if knn_indices is None:
            knn_indices = knn_data.get("indices")
        knn_indices = np.array(knn_indices)

    generator = OllamaGenerator(model=args.model, workers=args.workers)
    epoch_scores = []

    # 3. 多轮推理循环
    for epoch in range(args.epochs):
        print(f"\n开始第 {epoch+1}/{args.epochs} 轮推理 ({args.method.upper()})...")
        all_preds = []
        
        for start in tqdm(range(0, len(dev_df), args.batch_size)):
            end = min(start + args.batch_size, len(dev_df))
            batch_prompts = []
            
            for i in range(start, end):
                if args.method == 'kate':
                    idxs = knn_indices[i][:args.k]
                else:
                    # Random 模式：每轮使用不同种子，确保平均值有意义
                    random.seed(epoch * 1000 + i)
                    idxs = random.sample(all_train_idxs, args.k)
                
                examples = [build_example(train_df.iloc[j]['table'], train_df.iloc[j]['sentence']) for j in idxs]
                prompt = build_prompt(examples, dev_df.iloc[i]['table'])
                batch_prompts.append(prompt)

            preds = generator(batch_prompts)
            all_preds.extend(preds)

        current_bleu = compute_bleu_score(all_preds, dev_df['sentence'].tolist())
        epoch_scores.append(current_bleu)
        print(f"本轮 BLEU: {current_bleu:.4f}")

    # 4. 计算平均分
    mean_bleu = np.mean(epoch_scores)
    std_bleu = np.std(epoch_scores)

    # 5. 保存结果
    os.makedirs("result", exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M")
    model_tag = args.model.replace(":", "")
    res_path = f"result/ToTTo_{args.method}_{model_tag}_k{args.k}_summary.txt"
    
    with open(res_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model} | Method: {args.method} | Samples: {args.sample_size}\n")
        f.write(f"Epochs: {args.epochs} | K: {args.k}\n")
        f.write(f"Mean BLEU: {mean_bleu:.4f}\n")
        f.write(f"Std Dev: {std_bleu:.4f}\n")
        f.write(f"All Scores: {epoch_scores}\n")

    print(f"\n" + "="*40)
    print(f"评估完成！平均 BLEU: {mean_bleu:.4f} (±{std_bleu:.4f})")
    print(f"统计结果已保存至: {res_path}")
    print("="*40)

if __name__ == "__main__":
    main()