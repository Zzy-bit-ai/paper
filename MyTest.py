import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
import time
import string
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi

# ==================== 参数配置 ====================
parser = argparse.ArgumentParser(description="TriviaQA Retrieve + Rerank + Few-shot Generate (Fixed Version)")
parser.add_argument('--train_file', type=str, required=True, help="训练集 TSV（作为语料库）")
parser.add_argument('--dev_file', type=str, required=True, help="验证集 TSV（测试问题）")
parser.add_argument('--ranker_model', type=str, required=True, help="训练好的 ranker pth 权重路径")
parser.add_argument('--hf_model_path', type=str, default='gpt2', help="用于计算 PPL 的模型")
parser.add_argument('--llm_model', type=str, default='qwen:1.8b', help="Ollama 模型名称，例如 qwen:1.8b")
parser.add_argument('--rough_k', type=int, default=100, help="BM25 粗排候选数量（推荐 100+）")
parser.add_argument('--k_candidates', type=int, default=8, help="最终选为 Few-shot 示例的数量")
parser.add_argument('--max_samples', type=int, default=None, help="测试样本数量，None=全部")
parser.add_argument('--use_ranker', action='store_true', default=True, help="是否使用 ranker（关闭即为纯 BM25 基线）")
parser.add_argument('--no_use_ranker', dest='use_ranker', action='store_false', help="关闭 ranker，使用 BM25 基线")
parser.add_argument('--output_json', type=str, default='qa_results.json', help="详细结果保存路径")
parser.add_argument('--ollama_retries', type=int, default=3, help="Ollama 调用重试次数")
parser.add_argument('--ollama_timeout', type=int, default=60, help="Ollama 请求超时时间")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 与训练时完全一致的模型结构 ====================
class LocalRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),        # 训练时是 0.1
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_ranker(path):
    model = LocalRanker().to(device)
    # 使用 weights_only=True 避免安全警告并提高安全性
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model

# ==================== 辅助工具 ====================
def normalize_answer(s):
    s = str(s).lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())

def exact_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)

# 与训练时完全一致的特征缩放
def scale_features(ppl_val: float, bm25_score: float):
    return [np.log1p(ppl_val) / 5.0, bm25_score / 20.0]

# ==================== PPL 计算（批处理） ====================
print("🤖 加载 PPL 计算模型...")
tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ppl_model = AutoModelForCausalLM.from_pretrained(
    args.hf_model_path,
    torch_dtype=torch.float16
).to(device)
ppl_model.eval()

def compute_ppl_batch(prompts):
    if not prompts:
        return np.array([])
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = ppl_model(**inputs, labels=inputs["input_ids"])

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.shape)

    mask = (shift_labels != tokenizer.pad_token_id).float()
    seq_loss = (loss * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
    return torch.exp(seq_loss).cpu().numpy()

# ==================== Ollama 推理（带重试） ====================
def ollama_generate(prompt: str):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": args.llm_model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
        "options": {
            "num_predict": 64,
            "stop": ["\n", "\n\n", "Question:", "question:"]
        }
    }

    for attempt in range(args.ollama_retries):
        try:
            r = requests.post(url, json=payload, timeout=args.ollama_timeout)
            if r.status_code == 200:
                response = r.json().get("response", "").strip()
                if response:
                    return response
        except Exception as e:
            print(f"  ⚠️ Ollama 调用失败 (尝试 {attempt+1}/{args.ollama_retries}): {e}")
            time.sleep(2 ** attempt)
    return ""

# ==================== 主流程 ====================
def main():
    ranker = load_ranker(args.ranker_model) if args.use_ranker else None

    print("📂 加载数据...")
    train_df = pd.read_csv(args.train_file, sep='\t', quoting=3, on_bad_lines='skip').fillna("")
    dev_df   = pd.read_csv(args.dev_file,   sep='\t', quoting=3, on_bad_lines='skip').fillna("")

    test_df = dev_df.head(args.max_samples) if args.max_samples else dev_df
    print(f"测试样本数: {len(test_df)}")

    corpus_records = train_df.to_dict('records')
    print("🔍 构建 BM25 索引...")
    bm25 = BM25Okapi([str(r['q']).lower().split() for r in corpus_records])

    results = []
    correct_count = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="QA 处理"):
        q_test = str(row['q']).strip()
        gold   = str(row['a']).strip()

        # 1. BM25 粗排 Top-rough_k
        bm25_scores = bm25.get_scores(q_test.lower().split())
        top_indices = np.argsort(bm25_scores)[-args.rough_k:]

        cands = [corpus_records[i] for i in top_indices]
        bm25_subscores = [bm25_scores[i] for i in top_indices]

        # 2. 计算 PPL
        prompts = [f"Q: {c['q']} A: {c['a']}\nQ: {q_test} A:" for c in cands]
        ppls = compute_ppl_batch(prompts)

        # 3. 构造特征
        features = [scale_features(ppls[i], bm25_subscores[i]) for i in range(len(cands))]
        features_tensor = torch.tensor(features, dtype=torch.float).to(device)

        # 4. 重排序
        if args.use_ranker and ranker is not None:
            with torch.no_grad():
                rank_scores = ranker(features_tensor).cpu().numpy()
            sorted_indices = np.argsort(rank_scores)[::-1]  # 从高到低
        else:
            # 基线：直接按 BM25 分数（已降序）
            sorted_indices = list(range(len(cands)))

        # 5. 选取 Top-k 作为 Few-shot 示例
        selected_indices = sorted_indices[:args.k_candidates]
        selected_cands = [cands[i] for i in selected_indices]

        # 6. 构造清晰的 Prompt
        final_prompt = (
            "Use the following examples to answer the question. "
            "Provide only the answer, no explanation.\n\n"
        )
        for c in selected_cands:
            final_prompt += f"Question: {c['q']}\nAnswer: {c['a']}\n\n"
        final_prompt += f"Question: {q_test}\nAnswer:"

        # 7. 调用 Ollama 生成答案
        pred = ollama_generate(final_prompt)

        is_correct = exact_match(pred, gold)
        if is_correct:
            correct_count += 1

        results.append({
            "question": q_test,
            "prediction": pred,
            "gold": gold,
            "correct": is_correct,
            "method": "ranker" if args.use_ranker else "bm25_baseline"
        })

    # ==================== 结果输出 ====================
    accuracy = correct_count / len(results)
    print(f"\n✅ 测试完成！")
    print(f"   方法: {'Ranker + Few-shot' if args.use_ranker else 'BM25 基线 Few-shot'}")
    print(f"   Exact Match 准确率: {accuracy:.4f} ({correct_count}/{len(results)})")

    # 保存详细结果
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"   详细结果已保存至: {args.output_json}")

if __name__ == "__main__":
    main()