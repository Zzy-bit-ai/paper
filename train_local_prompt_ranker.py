# train_local_prompt_ranker_optimized.py
# 修复 I/O 瓶颈、优化答案匹配、增强负采样

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import json
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
import argparse
import re
import string
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== 参数配置 ====================
parser = argparse.ArgumentParser(description="BM25 + PPL Pairwise Ranker for TriviaQA (Optimized)")
parser.add_argument('--train_file', type=str, required=True, help="训练集 TSV 文件路径")
parser.add_argument('--dev_file', type=str, required=True, help="验证集 TSV 文件路径")
parser.add_argument('--cache_file', type=str, default='ppl_cache.jsonl', help="PPL 特征缓存文件")
parser.add_argument('--hf_model_path', type=str, default='gpt2', help="HuggingFace 语言模型路径")
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--k_candidates', type=int, default=60, help="BM25召回的候选数量")
parser.add_argument('--neg_ratio', type=int, default=5, help="每个正样本配对的负样本数量")
parser.add_argument('--train_nrows', type=int, default=None, help="训练集读取行数")
parser.add_argument('--dev_nrows', type=int, default=None, help="验证集读取行数")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--ppl_batch_size', type=int, default=32, help="PPL 计算批量大小")
parser.add_argument('--save_model_dir', type=str, default='./saved_models')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 1. 标准化与匹配工具 ====================
def normalize_answer(s):
    """标准的 TriviaQA/SQuAD 答案标准化"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def exact_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)

def check_containment(cand_ans, gold_ans):
    """检查候选答案是否包含金标答案（经过标准化）"""
    norm_cand = normalize_answer(cand_ans)
    norm_gold = normalize_answer(gold_ans)
    if not norm_gold: return False
    return norm_gold in norm_cand # 允许候选答案较长，包含正确答案

# ==================== 2. 优化的 FeatureCache ====================
class FeatureCache:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.data = {}
        self.write_buffer = []
        self.buffer_size = 500  # 缓冲区大小，减少IO次数

        if os.path.exists(cache_path):
            print(f"📦 加载 PPL 缓存文件: {cache_path}")
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            # 使用 tuple 作为 key
                            key = (item['query'], item['cand_q'], item['cand_a'])
                            self.data[key] = item['ppl']
                print(f"✅ 已加载 {len(self.data)} 条缓存记录")
            except Exception as e:
                print(f"⚠️ 缓存文件读取错误: {e}")

    def get(self, query, cand_q, cand_a):
        return self.data.get((query, cand_q, cand_a))

    def add_to_buffer(self, query, cand_q, cand_a, ppl):
        key = (query, cand_q, cand_a)
        if key not in self.data:
            self.data[key] = ppl
            self.write_buffer.append({'query': query, 'cand_q': cand_q, 'cand_a': cand_a, 'ppl': ppl})
        
        if len(self.write_buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.write_buffer:
            return
        with open(self.cache_path, 'a', encoding='utf-8') as f:
            for item in self.write_buffer:
                json.dump(item, f)
                f.write('\n')
        self.write_buffer = []

# ==================== 3. 语言模型加载 ====================
print(f"🤖 正在加载语言模型: {args.hf_model_path} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model_path,torch_dtype=torch.float16).to(device)
    hf_model.eval()
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# ==================== 4. 核心计算函数 ====================
def compute_ppl_batch(prompts: list[str]) -> list[float]:
    if not prompts:
        return []

    # 批处理编码
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = hf_model(**inputs, labels=inputs["input_ids"])

    # Shift logits and labels for causal LM loss
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token_loss = per_token_loss.view(shift_labels.size())

    # 计算有效长度 (忽略 padding)
    lengths = (shift_labels != tokenizer.pad_token_id).sum(dim=1).float()
    # 修复 ZeroDivisionError: 强制最小长度为 1.0
    lengths = torch.clamp(lengths, min=1.0)

    seq_loss = per_token_loss.sum(dim=1) / lengths
    return torch.exp(seq_loss).cpu().numpy().tolist()

def scale_features(ppl_val: float, bm25_score: float):
    # Log scale PPL because it can be very large
    # BM25 is usually 10-40 range
    return [np.log1p(ppl_val) / 5.0, bm25_score / 20.0]

# ==================== 5. 数据集构建 ====================
class RankingDataset(Dataset):
    def __init__(self, query_df, corpus_df, bm25_obj, cache_manager, desc):
        self.samples = []
        
        # 预先处理 corpus 数据以加快访问
        corpus_records = corpus_df.to_dict('records')
        
        for idx in tqdm(range(len(query_df)), desc=desc):
            row = query_df.iloc[idx]
            query_text = str(row['q']).strip()
            gold_ans = str(row['a']).strip()

            if not query_text or not gold_ans:
                continue

            # 1. BM25 粗排检索
            scores = bm25_obj.get_scores(query_text.lower().split())
            top_idx = np.argsort(scores)[-args.k_candidates:]

            candidates = []
            for c_idx in top_idx:
                c_row = corpus_records[c_idx]
                c_q = str(c_row['q']).strip()
                c_a = str(c_row['a']).strip()
                candidates.append({
                    'c_q': c_q,
                    'c_a': c_a,
                    'score': float(scores[c_idx])
                })

            # 2. 准备 PPL 计算任务
            prompts_to_compute = []
            compute_indices = [] # 记录需要计算的候选下标
            final_ppls = [None] * len(candidates)

            for i, cand in enumerate(candidates):
                cached = cache_manager.get(query_text, cand['c_q'], cand['c_a'])
                if cached is not None:
                    final_ppls[i] = cached
                else:
                    # 构造 Prompt: Q: .. A: .. \n Q: .. A:
                    prompt = f"Q: {cand['c_q']} A: {cand['c_a']}\nQ: {query_text} A:"
                    prompts_to_compute.append(prompt)
                    compute_indices.append(i)

            # 3. 批量计算缺失的 PPL
            if prompts_to_compute:
                new_ppls = []
                for start in range(0, len(prompts_to_compute), args.ppl_batch_size):
                    batch = prompts_to_compute[start : start + args.ppl_batch_size]
                    batch_res = compute_ppl_batch(batch)
                    new_ppls.extend(batch_res)
                
                # 回填并加入缓存
                for i, idx_in_cand in enumerate(compute_indices):
                    ppl_val = new_ppls[i]
                    final_ppls[idx_in_cand] = ppl_val
                    cand = candidates[idx_in_cand]
                    cache_manager.add_to_buffer(query_text, cand['c_q'], cand['c_a'], ppl_val)

            # 4. 生成训练对 (Pairwise)
            pos_feats = []
            neg_feats = []

            for i, cand in enumerate(candidates):
                feat = torch.tensor(scale_features(final_ppls[i], cand['score']), dtype=torch.float)
                
                # 使用严格的匹配逻辑
                is_correct = check_containment(cand['c_a'], gold_ans)
                
                if is_correct:
                    pos_feats.append(feat)
                else:
                    neg_feats.append(feat)

            # 改进采样策略：1个正样本配对 N 个负样本
            if pos_feats and neg_feats:
                for pos_feat in pos_feats:
                    # 如果负样本不够，就有多少取多少
                    current_neg_count = min(len(neg_feats), args.neg_ratio)
                    chosen_negs = random.sample(neg_feats, current_neg_count)
                    for neg_feat in chosen_negs:
                        self.samples.append((pos_feat, neg_feat))
        
        # 处理完所有数据后，强制刷新缓存写入
        cache_manager.flush()
        print(f"✅ {desc} 完成，共生成 {len(self.samples)} 个训练对")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==================== 6. 模型定义 ====================
class LocalRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1), # 防止过拟合
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ==================== 7. 评估逻辑 ====================
@torch.no_grad()
def evaluate_ranking_metrics(ranker, query_df, corpus_df, bm25_obj, cache_manager, k=60):
    ranker.eval()
    reciprocal_ranks = []
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    
    corpus_records = corpus_df.to_dict('records')

    for idx in tqdm(range(len(query_df)), desc="📊 评估中", leave=False):
        row = query_df.iloc[idx]
        query_text = str(row['q']).strip()
        gold_ans = str(row['a']).strip()

        scores_bm25 = bm25_obj.get_scores(query_text.lower().split())
        top_idx = np.argsort(scores_bm25)[-k:]

        features = []
        is_correct_list = []

        # 收集特征（评估时如果缓存没有PPL，这里简化处理跳过计算，或者你需要在这里也加上PPL计算逻辑）
        # 假设训练阶段缓存已经覆盖大部分，或者这里接受少部分缺失
        # 为了严谨，这里简单实现实时计算（单条较慢，但评估集通常较小）
        
        prompts = []
        cand_indices = []
        
        temp_candidates = []

        for c_idx in top_idx:
            c_row = corpus_records[c_idx]
            c_q = str(c_row['q']).strip()
            c_a = str(c_row['a']).strip()
            
            ppl = cache_manager.get(query_text, c_q, c_a)
            if ppl is None:
                # 评估时也需要计算PPL
                prompt = f"Q: {c_q} A: {c_a}\nQ: {query_text} A:"
                prompts.append(prompt)
                cand_indices.append(len(temp_candidates))
                ppl_val = 0.0 # 占位
            else:
                ppl_val = ppl
            
            temp_candidates.append({
                'score': float(scores_bm25[c_idx]),
                'ppl': ppl_val,
                'is_correct': check_containment(c_a, gold_ans),
                'c_q': c_q,
                'c_a': c_a
            })

        # 补算 PPL
        if prompts:
            # 分批算
            for start in range(0, len(prompts), args.ppl_batch_size):
                batch = prompts[start:start+args.ppl_batch_size]
                res = compute_ppl_batch(batch)
                # 填回
                for b_idx, r_ppl in enumerate(res):
                    real_idx = cand_indices[start + b_idx]
                    temp_candidates[real_idx]['ppl'] = r_ppl
                    # 存入缓存以备后用
                    c = temp_candidates[real_idx]
                    cache_manager.add_to_buffer(query_text, c['c_q'], c['c_a'], r_ppl)
            cache_manager.flush()

        # 构建 Tensor
        for cand in temp_candidates:
            features.append(scale_features(cand['ppl'], cand['score']))
            is_correct_list.append(cand['is_correct'])

        if not features:
            reciprocal_ranks.append(0.0)
            continue

        features_tensor = torch.tensor(features, dtype=torch.float).to(device)
        model_scores = ranker(features_tensor).cpu().numpy()
        
        # 从高到低排序的索引
        ranked_order = np.argsort(model_scores)[::-1]

        # 寻找第一个正确答案的排名
        best_rank = None
        for rank, pos_idx in enumerate(ranked_order, 1):
            if is_correct_list[pos_idx]:
                best_rank = rank
                break

        if best_rank:
            reciprocal_ranks.append(1.0 / best_rank)
            for topn in hits:
                if best_rank <= topn:
                    hits[topn] += 1
        else:
            reciprocal_ranks.append(0.0)

    total = len(reciprocal_ranks) if reciprocal_ranks else 1
    mrr_10 = np.mean([rr if rr >= 0.1 else 0.0 for rr in reciprocal_ranks]) # 1/10 = 0.1
    mrr_60 = np.mean(reciprocal_ranks)
    recall = {f"Recall@{topn}": count / total for topn, count in hits.items()}

    return {
        "MRR@10": mrr_10,
        "MRR@60": mrr_60,
        **recall
    }

# ==================== 8. 主程序 ====================
def main():
    cache_manager = FeatureCache(args.cache_file)

    print("📂 读取数据文件...")
    # 注意：quoting=3 用于处理可能的引号问题
    train_df = pd.read_csv(args.train_file, sep='\t', nrows=args.train_nrows, quoting=3, on_bad_lines='skip').fillna("")
    dev_df   = pd.read_csv(args.dev_file,   sep='\t', nrows=args.dev_nrows,   quoting=3, on_bad_lines='skip').fillna("")
    
    print(f"   Train Size: {len(train_df)}")
    print(f"   Dev Size:   {len(dev_df)}")

    print("🔍 构建 BM25 索引...")
    # 仅使用 Training Set 构建索引语料库
    bm25 = BM25Okapi([str(q).lower().split() for q in train_df['q']])

    print("🛠️ 构造训练集...")
    train_ds = RankingDataset(train_df, train_df, bm25, cache_manager, desc="训练集处理")
    
    # 验证集 Dataset 仅用于 loss 计算，Metric 评估单独跑
    print("🛠️ 构造验证集 (用于 Loss)...")
    val_ds   = RankingDataset(dev_df, train_df, bm25, cache_manager, desc="验证集处理")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    ranker = LocalRanker().to(device)
    optimizer = optim.Adam(ranker.parameters(), lr=args.lr)
    criterion = nn.MarginRankingLoss(margin=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    best_mrr10 = 0.0
    train_losses, val_losses = [], []
    mrr10_history = []

    os.makedirs(args.save_model_dir, exist_ok=True)

    print("\n🚀 开始训练...")
    for epoch in range(args.epochs):
        ranker.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for pos, neg in pbar:
            pos, neg = pos.to(device), neg.to(device)
            optimizer.zero_grad()
            
            score_pos = ranker(pos)
            score_neg = ranker(neg)
            
            # Label=1 means pos should be higher than neg
            target = torch.ones_like(score_pos)
            loss = criterion(score_pos, score_neg, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_losses.append(avg_train_loss)

        # 验证 Loss
        ranker.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pos, neg in val_loader:
                pos, neg = pos.to(device), neg.to(device)
                score_pos = ranker(pos)
                score_neg = ranker(neg)
                loss = criterion(score_pos, score_neg, torch.ones_like(score_pos))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)

        # 验证 Ranking Metrics
        metrics = evaluate_ranking_metrics(ranker, dev_df, train_df, bm25, cache_manager, k=args.k_candidates)
        mrr10 = metrics["MRR@10"]
        mrr10_history.append(mrr10)

        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   MRR@10: {mrr10:.4f} (Best: {max(best_mrr10, mrr10):.4f})")
        print(f"   R@1: {metrics['Recall@1']:.4f} | R@5: {metrics['Recall@5']:.4f}")

        if mrr10 > best_mrr10:
            best_mrr10 = mrr10
            save_path = os.path.join(args.save_model_dir, f"best_ranker_mrr{mrr10:.3f}.pth")
            torch.save(ranker.state_dict(), save_path)
            print(f"💾 模型已保存: {save_path}")

        scheduler.step(mrr10)

    # 绘制曲线
    try:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(train_losses, label="Train Loss", color="tab:blue")
        ax1.plot(val_losses, label="Val Loss", color="tab:orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper left")
        
        ax2 = ax1.twinx()
        ax2.plot(mrr10_history, label="MRR@10", color="tab:green", marker='o', linestyle='--')
        ax2.set_ylabel("MRR@10")
        ax2.legend(loc="upper right")
        
        plt.title("Training Loss & MRR@10")
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_model_dir, "training_curve.png"))
        print("📈 训练曲线已保存。")
    except Exception as e:
        print(f"绘图失败: {e}")

if __name__ == "__main__":
    main()