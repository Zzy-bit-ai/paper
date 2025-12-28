# test_bm25_baseline.py
import torch
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import requests
import re
import string
from tqdm import tqdm
import time

# ================= 配置 =================
TRAIN_FILE = r"D:\py_code\paper\KATEGPT3-main\KATEGPT3-main\inference\dataset\trivia_qa_train_78785_dev_full_train.tsv"
TEST_SIZE = 500 
OLLAMA_MODEL = "qwen:1.8b" # 用稍微大一点的这个测
K_SHOT = 3

# ================= 工具 =================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def load_data(path):
    print(f"📖 读取数据: {path}")
    try:
        df = pd.read_csv(path, sep='\t', quoting=3, on_bad_lines='skip').fillna("")
        if len(df.columns) < 2: df = pd.read_csv(path, sep=',', quoting=3, on_bad_lines='skip').fillna("")
    except:
        df = pd.read_csv(path, sep=None, engine='python', quoting=3, on_bad_lines='skip').fillna("")
    col_map = {}
    for c in df.columns:
        if str(c).lower().strip() in ['q', 'question']: col_map[c] = 'q'
        if str(c).lower().strip() in ['a', 'answer']: col_map[c] = 'a'
    df.rename(columns=col_map, inplace=True)
    if 'q' not in df.columns: df['q'] = df.iloc[:,0]
    if 'a' not in df.columns: df['a'] = df.iloc[:,1]
    return df

def query_ollama(prompt):
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "stop": ["\n", "Q:"]}
        })
        return res.json().get("response", "").strip()
    except: return ""

# ================= 主流程 =================
def main():
    full_df = load_data(TRAIN_FILE)
    if TEST_SIZE: test_df = full_df.iloc[:TEST_SIZE].copy()
    else: test_df = full_df.iloc[-100:].copy()
    
    print("构建 BM25 索引...")
    bm25 = BM25Okapi([str(t).lower().split() for t in full_df['q']])
    corpus_records = full_df.to_dict('records')
    
    correct = 0
    total = 0
    
    print(f"🚀 开始纯 BM25 基线测试 ({OLLAMA_MODEL})...")
    
    with open("result_bm25_baseline.txt", "w", encoding="utf-8") as f:
        for idx in tqdm(range(len(test_df))):
            row = test_df.iloc[idx]
            q_text = str(row['q']).strip()
            gold_ans = str(row['a']).strip()
            
            # --- 纯 BM25 逻辑 ---
            scores = bm25.get_scores(q_text.lower().split())
            # 直接取分数最高的 K 个 (排除自己)
            top_idx = np.argsort(scores)[-60:]
            
            demos = []
            for c_idx in reversed(top_idx): # 从高到低遍历
                c_row = corpus_records[c_idx]
                if str(c_row['q']).strip() == q_text: continue
                demos.append(f"Q: {c_row['q']}\nA: {c_row['a']}")
                if len(demos) >= K_SHOT: break
            
            # 组装 (倒序：最相关的放在最后，靠近 Query)
            context = "\n\n".join(demos[::-1])
            final_input = f"{context}\n\nQ: {q_text}\nA:"
            
            # 推理
            pred = query_ollama(final_input)
            
            is_match = (normalize_answer(pred) == normalize_answer(gold_ans)) or (normalize_answer(gold_ans) in normalize_answer(pred))
            if is_match: correct += 1
            total += 1
            
            f.write(f"Q: {q_text}\nGold: {gold_ans} | Pred: {pred} | {is_match}\n\n")
            
    print(f"📊 BM25 Baseline: {correct}/{total} = {correct/total:.4f}")

if __name__ == "__main__":
    main()
