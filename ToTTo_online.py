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
import time
import nltk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dotenv import load_dotenv

# 加载 .env 配置文件
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 解决 Windows 乱码
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ==================== MiMo API 调用封装 ====================
def call_mimo_api(prompt, model="xiaomi/mimo-v2-flash:free"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
        "stop": ["\n", "Table:", "Description:"]
    }
    
    for attempt in range(3):  # 失败重试 3 次
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"⚠️ API Error {r.status_code}: {r.text}")
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return "[ERROR]"

# ==================== BLEU 评估逻辑 ====================
def compute_bleu(preds, refs):
    smoother = SmoothingFunction().method1
    bleus = []
    for p, r in zip(preds, refs):
        if p == "[ERROR]" or not p.strip():
            bleus.append(0.0)
            continue
        pred_tokens = nltk.word_tokenize(p.lower())
        ref_tokens = nltk.word_tokenize(r.lower())
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
        bleus.append(score)
    return np.mean(bleus) * 100

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="ToTTo 数据集路径")
    parser.add_argument('--knn_file', type=str, default=None, help="kNN 索引文件路径")
    parser.add_argument('--k', type=int, default=8, help="Few-shot 数量")
    parser.add_argument('--model', type=str, default="xiaomi/mimo-v2-flash:free")
    parser.add_argument('--method', type=str, default='kate', choices=['random', 'kate'])
    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--workers', type=int, default=10, help="API 并发请求数")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("❌ 错误: 未找到 API KEY，请在 .env 中配置")
        return

    # 1. 加载数据
    dev_df = pd.read_csv(os.path.join(args.data_dir, "ToTTo_dev.tsv"), sep='\t').fillna("").head(args.sample_size)
    train_df = pd.read_csv(os.path.join(args.data_dir, "ToTTo_train.tsv"), sep='\t').fillna("")
    
    knn_indices = None
    if args.method == 'kate':
        with open(args.knn_file, "rb") as f:
            knn_data = pickle.load(f)
        knn_indices = knn_data.get("kNN_dev_train") if knn_data.get("kNN_dev_train") is not None else knn_data.get("indices")
        knn_indices = np.array(knn_indices)

    all_epoch_results = []

    # 2. 实验循环
    for ep in range(args.epochs):
        print(f"\n🚀 开始 Epoch {ep+1}/{args.epochs} ({args.method.upper()})")
        prompts = []
        for i in range(len(dev_df)):
            if args.method == 'kate':
                idxs = knn_indices[i][:args.k]
            else:
                random.seed(ep * 1000 + i)
                idxs = random.sample(range(len(train_df)), args.k)
            
            p = "Instruction: Convert the following table into a fluent sentence.\n\n"
            for idx in idxs:
                p += f"Table: {train_df.iloc[idx]['table']}\nSent: {train_df.iloc[idx]['sentence']}\n\n"
            p += f"Table: {dev_df.iloc[i]['table']}\nSent:"
            prompts.append(p)

        # 并发调用 API
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # 使用 map 保证结果顺序
            preds = list(tqdm(executor.map(lambda pr: call_mimo_api(pr, args.model), prompts), 
                              total=len(prompts), desc="Calling MiMo API"))
        
        score = compute_bleu(preds, dev_df['sentence'].tolist())
        all_epoch_results.append(score)
        print(f"✅ Epoch {ep+1} BLEU: {score:.4f}")

    # 3. 统计与保存
    mean_s = np.mean(all_epoch_results)
    std_s = np.std(all_epoch_results)
    
    os.makedirs("result", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_model = args.model.replace("/", "_").replace(":", "_")
    res_file = f"result/ToTTo_Online_{args.method}_{safe_model}_k{args.k}_{timestamp}.txt"
    
    with open(res_file, "w", encoding="utf-8") as f:
        f.write(f"Results for {args.method} (k={args.k}, samples={args.sample_size})\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Mean BLEU: {mean_s:.4f} ± {std_s:.4f}\n")
        f.write(f"Epoch Scores: {all_epoch_results}\n")

    print(f"\n🏆 最终结果: 平均 BLEU = {mean_s:.2f} (±{std_s:.2f})")
    print(f"💾 结果已保存至: {res_file}")

if __name__ == "__main__":
    main()