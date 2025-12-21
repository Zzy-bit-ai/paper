# autocomplete_run_final.py
# KATE vs Random Baseline for TriviaQA - Ultimate Robust Version
# 专为你的 kNN 生成脚本优化：支持 {"kNN_dev_train": np.ndarray(shape=(n_dev, n_neighbors))}

import os
import sys
import pandas as pd
import numpy as np
import random
import time
import pickle
import argparse
import re
import string
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# ==================== 环境加载 ====================
load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")

if not OLLAMA_URL:
    print("❌ Error: OLLAMA_URL not set!")
    sys.exit(1)

print(f"✅ Ollama URL: {OLLAMA_URL}")

# ==================== 参数 ====================
parser = argparse.ArgumentParser(description="KATE/Random Baseline for TriviaQA")
parser.add_argument('--task_name', default='trivia_qa_train_78785_dev_full', type=str)
parser.add_argument('--knn_num', default=10, type=int, help="Number of in-context examples")
parser.add_argument('--max_tokens', default=32, type=int)
parser.add_argument('--kNN_dev_train', default='', type=str, help="kNN .dat filename without extension")
parser.add_argument('--random', action='store_true', help="Random baseline")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model', type=str, default='qwen:1.8b')
parser.add_argument('--max_workers', type=int, default=4, help="Concurrency (recommend 2-4)")
parser.add_argument('--timeout', type=int, default=120)
parser.add_argument('--base_path', default=r'D:\py_code\paper\KATEGPT3-main\KATEGPT3-main\inference', type=str)
parser.add_argument('--shuffle_examples', action='store_true', help="Shuffle examples to reduce order bias")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

mode_name = 'random' if args.random else 'kate'
print(f"🚀 Start | Model: {args.model} | Mode: {mode_name.upper()} | k={args.knn_num} | Workers: {args.max_workers}")

# ==================== 答案标准化 ====================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))

def exact_match_multi(pred, gold_list):
    pred_norm = normalize_answer(pred)
    return any(pred_norm == normalize_answer(g) for g in gold_list)

# ==================== Prompt 构建 ====================
def construct_prompt(train_df, indices, q_test):
    prompt = "Answer the following questions with a short phrase or name.\n\n"
    chosen = indices[:args.knn_num]

    if args.shuffle_examples:
        random.shuffle(chosen)

    for idx in chosen:
        row = train_df.iloc[idx]
        q = str(row['q']).strip().replace('\n', ' ').replace('\r', ' ')
        a = str(row['a']).strip().replace('\n', ' ').replace('\r', ' ')
        prompt += f"Question: {q}\nAnswer: {a}\n\n"

    q_clean = q_test.strip().replace('\n', ' ').replace('\r', ' ')
    prompt += f"Question: {q_clean}\nAnswer:"
    return prompt

# ==================== Ollama 调用 ====================
def ollama_generate(prompt, model):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
        "options": {
            "num_predict": args.max_tokens,
            "num_ctx": 8192,
            "stop": ["\n", "\n\n", "Question:", "Q:", "question:"]
        }
    }

    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=args.timeout)
            r.raise_for_status()
            resp = r.json().get("response", "").strip()
            if resp:
                return resp
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Ollama attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    print("❌ Ollama failed after 3 attempts")
    return None

# ==================== 数据加载 ====================
train_file = os.path.join(args.base_path, "dataset", f"{args.task_name}_train.tsv")
dev_file = os.path.join(args.base_path, "dataset", f"{args.task_name}_dev.tsv")

print(f"📂 Loading data...")
try:
    train_df = pd.read_csv(train_file, sep='\t', header=None, names=['q', 'a'], quoting=3, on_bad_lines='skip')
    dev_df = pd.read_csv(dev_file, sep='\t', header=None, names=['q', 'a'], quoting=3, on_bad_lines='skip')
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
except Exception as e:
    print(f"❌ Data load error: {e}")
    sys.exit(1)

print(f"📊 Train: {len(train_df)} | Dev: {len(dev_df)}")

valid_train_indices = [i for i in range(len(train_df))
                       if pd.notna(train_df.iloc[i]['q']) and pd.notna(train_df.iloc[i]['a'])
                       and str(train_df.iloc[i]['q']).strip() and str(train_df.iloc[i]['a']).strip()]
print(f"✅ Valid train examples: {len(valid_train_indices)} / {len(train_df)}")

# ==================== kNN 加载（关键：适配你的生成脚本） ====================
knn_indices_map = None  # 将统一为 list of lists
if not args.random:
    if not args.kNN_dev_train:
        print("❌ KATE mode needs --kNN_dev_train")
        sys.exit(1)

    knn_path = os.path.join(args.base_path, "kNN_pretraining", f"{args.kNN_dev_train}.dat")
    if not os.path.exists(knn_path):
        print(f"❌ kNN file not found: {knn_path}")
        sys.exit(1)

    try:
        with open(knn_path, "rb") as f:
            data = pickle.load(f)

        print(f"Raw kNN type: {type(data)}")

        # 提取 kNN_dev_train
        knn_array = data.get('kNN_dev_train') if isinstance(data, dict) else data

        if knn_array is None:
            print("❌ 'kNN_dev_train' not found")
            sys.exit(1)

        print(f"Extracted array type: {type(knn_array)} | shape: {getattr(knn_array, 'shape', 'N/A')}")

        if isinstance(knn_array, np.ndarray):
            if knn_array.ndim != 2:
                print(f"❌ Bad shape: {knn_array.shape}")
                sys.exit(1)
            knn_indices_map = knn_array.tolist()
        elif isinstance(knn_array, list):
            knn_indices_map = knn_array
        else:
            print(f"❌ Unsupported type: {type(knn_array)}")
            sys.exit(1)

        num_dev = len(knn_indices_map)
        neighbors = len(knn_indices_map[0]) if num_dev > 0 else 0
        print(f"✅ Loaded kNN: {num_dev} dev samples × {neighbors} neighbors")

    except Exception as e:
        print(f"❌ kNN load error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ==================== 构建任务 ====================
tasks = []
skipped = 0
print("📝 Building prompts...")
for i in range(len(dev_df)):
    q_test = str(dev_df.iloc[i]['q']).strip()
    gold_raw = dev_df.iloc[i]['a']

    if not q_test or pd.isna(gold_raw) or not str(gold_raw).strip():
        skipped += 1
        continue

    gold_list = [str(gold_raw).strip()]

    if args.random:
        chosen = random.sample(valid_train_indices, min(args.knn_num, len(valid_train_indices)))
    else:
        indices = knn_indices_map[i] if i < len(knn_indices_map) else []
        indices = [int(idx) for idx in indices if isinstance(idx, (int, np.int64)) 
                  and 0 <= idx < len(train_df) and idx in valid_train_indices]
        chosen = indices

    # 补充到 k 个
    if len(chosen) < args.knn_num:
        needed = args.knn_num - len(chosen)
        pool = [idx for idx in valid_train_indices if idx not in set(chosen)]
        extra = random.sample(pool, min(needed, len(pool))) if pool else random.choices(valid_train_indices, k=needed)
        chosen.extend(extra)

    chosen = chosen[:args.knn_num]

    prompt = construct_prompt(train_df, chosen, q_test)
    tasks.append({"id": i, "prompt": prompt, "gold_list": gold_list})

print(f"⚠️ Skipped {skipped} invalid samples")
print(f"✅ {len(tasks)} valid tasks ready")

# ==================== 推理 ====================
print(f"🚀 Inferring ({args.max_workers} workers)...")
results = []
correct_count = 0
error_count = 0

with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    future_to_task = {executor.submit(ollama_generate, t['prompt'], args.model): t for t in tasks}
    pbar = tqdm(as_completed(future_to_task), total=len(tasks), desc="Inferring")

    for future in pbar:
        task = future_to_task[future]
        try:
            pred = future.result()
            if pred is None:
                pred = "[OLLAMA FAILED]"
                error_count += 1
                is_correct = False
            else:
                is_correct = exact_match_multi(pred, task['gold_list'])

            results.append({"id": task['id'], "gold": task['gold_list'], "pred": pred, "correct": is_correct})
            if is_correct:
                correct_count += 1

            pbar.set_postfix({"acc": f"{correct_count/len(results):.4f}", "err": error_count})

        except Exception as e:
            print(f"\n❌ Task {task['id']} error: {e}")
            error_count += 1

# ==================== 结果 ====================
acc = correct_count / len(results) if results else 0.0
print(f"\n🏆 Final Accuracy: {acc:.4f} ({correct_count}/{len(results)})")
print(f"   Ollama errors: {error_count}")

# 保存
safe_model = args.model.replace(":", "_").replace("/", "_")
output_file = f"result_{safe_model}_{mode_name}_k{args.knn_num}_acc{acc:.4f}.pkl"
output_path = os.path.join(args.base_path, output_file)

with open(output_path, "wb") as f:
    pickle.dump({
        "args": vars(args),
        "accuracy": acc,
        "correct": correct_count,
        "total": len(results),
        "errors": error_count,
        "results": results
    }, f)

print(f"💾 Saved to: {output_path}")