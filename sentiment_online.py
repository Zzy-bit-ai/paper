# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import random
import time
import pickle
import argparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

# ==================== 环境加载 ====================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ==================== 参数配置 ====================
parser = argparse.ArgumentParser(description="KATE 情感分类在线评估 - 优化版")
parser.add_argument('--data_dir', type=str, default='./dataset', help='数据集所在目录')
parser.add_argument('--train_file', type=str, default='SST-2_train.tsv')
parser.add_argument('--dev_file', type=str, default='IMDB_dev.tsv')
parser.add_argument('--knn_file', type=str, default=None, help='kNN 索引文件路径')
parser.add_argument('--k', type=int, default=8, help='示例数量')
parser.add_argument('--sample_size', type=int, default=2000, help='评估样本总数')
parser.add_argument('--epochs', type=int, default=3, help='重复运行次数')
parser.add_argument('--mode', type=str, default='kate', choices=['random', 'kate'])
parser.add_argument('--model', type=str, default='xiaomi/mimo-v2-flash:free')
parser.add_argument('--max_workers', type=int, default=10, help='API 并发数')
args = parser.parse_args()

# ==================== 功能组件 ====================

def call_mimo_api(prompt):
    """API 调用：增加明确的超时和重试逻辑"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 15
    }
    
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return content.strip().lower()
            else:
                print(f"⚠️ API Error {response.status_code}: {response.text}")
        except Exception as e:
            if attempt < 2: time.sleep(2 * (attempt + 1))
    return "[API_ERROR]"

def is_correct(pred, gold_label):
    """标签判断：更加严谨的逻辑"""
    pred = pred.lower()
    # 过滤掉干扰性前缀，如 "the sentiment is positive"
    if gold_label == 1: # Positive
        return "positive" in pred and "not positive" not in pred
    else: # Negative
        return "negative" in pred and "not negative" not in pred

def normalize_data(df):
    """数据标准化"""
    label_col = 'label' if 'label' in df.columns else 'Sentiment'
    df['std_label'] = df[label_col].apply(lambda x: 1 if str(x).lower() in ['1', 'positive', 'pos'] else 0)
    for col in ['sentence', 'text', 'Sentence', 'Text']:
        if col in df.columns:
            df = df.rename(columns={col: 'std_text'})
            break
    return df

# ==================== 主程序逻辑 ====================

def main():
    if not OPENROUTER_API_KEY:
        print("❌ 错误: 未找到 API KEY，请检查 .env 文件。")
        return

    # 1. 加载数据
    try:
        train_df = normalize_data(pd.read_csv(os.path.join(args.data_dir, args.train_file), sep='\t'))
        dev_full = normalize_data(pd.read_csv(os.path.join(args.data_dir, args.dev_file), sep='\t'))
        print(f"✅ 数据加载成功。训练集: {len(train_df)}, 验证集: {len(dev_full)}")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return

    # 2. 加载 KNN
    knn_indices = None
    if args.mode == 'kate':
        try:
            with open(args.knn_file, "rb") as f:
                knn_data = pickle.load(f)
            knn_indices = knn_data.get("kNN_dev_train") if knn_data.get("kNN_dev_train") is not None else knn_data.get("indices")
            knn_indices = np.array(knn_indices)
            print(f"✅ kNN 索引加载成功，Shape: {knn_indices.shape}")
        except Exception as e:
            print(f"❌ kNN 文件加载失败: {e}")
            return

    results_across_epochs = []

    # 3. Epoch 循环
    for ep in range(args.epochs):
        print(f"\n--- Epoch {ep+1}/{args.epochs} ({args.mode.upper()}) ---")
        
        # 样本洗牌与抽样（针对 Random 模式增加实验多样性）
        if args.mode == 'random':
            current_dev = dev_full.sample(n=min(args.sample_size, len(dev_full)), random_state=ep).reset_index()
        else:
            current_dev = dev_full.head(args.sample_size).reset_index()

        prompts = []
        gold_labels = []

        for i in range(len(current_dev)):
            # 获取原始索引以对齐 kNN
            original_idx = current_dev.iloc[i]['index']
            
            if args.mode == 'kate':
                idxs = knn_indices[original_idx][:args.k]
            else:
                random.seed(ep * 1000 + i)
                idxs = random.sample(range(len(train_df)), args.k)
            
            # 构建 Prompt
            p = "Instruction: Classify the following movie review as positive or negative.\n\n"
            for tidx in idxs:
                row = train_df.iloc[tidx]
                lbl = "positive" if row['std_label'] == 1 else "negative"
                p += f"Review: {row['std_text']}\nSentiment: {lbl}\n\n"
            
            p += f"Review: {current_dev.iloc[i]['std_text']}\nSentiment:"
            prompts.append(p)
            gold_labels.append(current_dev.iloc[i]['std_label'])

        # 4. 并发推理与结果对齐
        predictions = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 使用字典建立 future 与原始索引的映射
            future_to_idx = {executor.submit(call_mimo_api, p): i for i, p in enumerate(prompts)}
            
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Inference"):
                idx = future_to_idx[future]
                try:
                    predictions[idx] = future.result()
                except Exception as e:
                    predictions[idx] = "[ERROR]"

        # 5. 计算分数
        correct = sum(1 for p, g in zip(predictions, gold_labels) if is_correct(p, g))
        acc = (correct / len(gold_labels)) * 100
        results_across_epochs.append(acc)
        print(f"⭐ Epoch {ep+1} Accuracy: {acc:.2f}%")

    # 6. 最终统计
    mean_acc = np.mean(results_across_epochs)
    std_acc = np.std(results_across_epochs)
    
    print(f"\n{'='*50}")
    print(f"🏆 Final Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}")
    print(f"Details: {results_across_epochs}")
    print(f"{'='*50}")

    # 保存
    os.makedirs("result", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"result/Sent_{args.mode}_k{args.k}_{ts}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\nMode: {args.mode}\nK: {args.k}\n")
        f.write(f"Mean: {mean_acc:.4f}\nStd: {std_acc:.4f}\nAll: {results_across_epochs}")

if __name__ == "__main__":
    main()