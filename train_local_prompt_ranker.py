# train_ranker_v3_final.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import argparse
import re
import string
import hashlib
import sqlite3
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置 =================
REAL_DATA_PATH = r"D:\py_code\paper\KATEGPT3-main\KATEGPT3-main\inference\dataset\trivia_qa_train_78785_dev_full_train.tsv"
SAVE_DIR = "./processed_data_v3"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GPT2_MODEL_NAME = 'gpt2'
DB_PATH = "features_cache_v3.db"

# ================= 参数 =================
parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default=REAL_DATA_PATH)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--k_candidates', type=int, default=50) # 扩大候选池
parser.add_argument('--neg_ratio', type=int, default=5)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 工具函数 =================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def is_positive_sample(cand_ans, gold_ans):
    """[修复 #1] 双向包含或相等"""
    nc = normalize_answer(cand_ans)
    ng = normalize_answer(gold_ans)
    if not nc or not ng: return False
    return (nc == ng) or (nc in ng) or (ng in nc)

def get_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def smart_read_csv(file_path):
    print(f"📖 读取文件: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='skip').fillna("")
        if len(df.columns) < 2: df = pd.read_csv(file_path, sep=',', quoting=3, on_bad_lines='skip').fillna("")
    except:
        df = pd.read_csv(file_path, sep=None, engine='python', quoting=3, on_bad_lines='skip').fillna("")
    
    col_map = {}
    for c in df.columns:
        if str(c).lower().strip() in ['q', 'question']: col_map[c] = 'q'
        if str(c).lower().strip() in ['a', 'answer']: col_map[c] = 'a'
    df.rename(columns=col_map, inplace=True)
    if 'q' not in df.columns: df['q'] = df.iloc[:,0]
    if 'a' not in df.columns: df['a'] = df.iloc[:,1]
    return df

# ================= PPL 计算 (带缓存) =================
class PPLEngine:
    def __init__(self, model_path, db_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        self.model.eval()
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS ppl (hash TEXT PRIMARY KEY, val REAL)')
        self.conn.commit()
    
    def get_ppl_batch(self, prompts, hashes):
        # 1. 查缓存
        results = {}
        missing_indices = []
        missing_prompts = []
        
        # 批量查库
        if not hashes: return []
        
        # SQLite 限制，分批查
        CHUNK = 900
        for i in range(0, len(hashes), CHUNK):
            sub_h = hashes[i:i+CHUNK]
            p_str = ','.join(['?']*len(sub_h))
            self.cursor.execute(f"SELECT hash, val FROM ppl WHERE hash IN ({p_str})", sub_h)
            results.update({r[0]: r[1] for r in self.cursor.fetchall()})
            
        final_ppls = []
        for i, h in enumerate(hashes):
            if h in results:
                final_ppls.append(results[h])
            else:
                final_ppls.append(None)
                missing_indices.append(i)
                missing_prompts.append(prompts[i])
        
        # 2. 计算缺失
        if missing_prompts:
            # 分批推理
            BS = 32
            new_vals = []
            for j in range(0, len(missing_prompts), BS):
                batch_p = missing_prompts[j:j+BS]
                inputs = self.tokenizer(batch_p, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                
                # Shift logits
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size())
                
                # Mean per sequence
                lengths = (shift_labels != self.tokenizer.pad_token_id).sum(dim=1).float().clamp(min=1.0)
                seq_loss = loss.sum(dim=1) / lengths
                batch_ppl = torch.exp(seq_loss.clamp(max=15.0)).cpu().numpy().tolist() # max=15防止溢出
                new_vals.extend(batch_ppl)
            
            # 3. 写入缓存并填回
            rows = []
            for k, val in enumerate(new_vals):
                origin_idx = missing_indices[k]
                final_ppls[origin_idx] = val
                rows.append((hashes[origin_idx], val))
            
            self.cursor.executemany("INSERT OR IGNORE INTO ppl VALUES (?, ?)", rows)
            self.conn.commit()
            
        return final_ppls

    def close(self): self.conn.close()

# ================= 特征处理 =================
def build_dataset(query_df, corpus_df, bm25, sbert, ppl_engine, name):
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{name}_v3.pt")
    
    if os.path.exists(save_path):
        print(f"📦 发现缓存 {name}, 直接加载...")
        return torch.load(save_path)

    print(f"⚙️ 开始处理 {name} (3-Feature Mode)...")
    
    # 1. 预计算 Corpus Embedding
    print("   计算 Corpus Embeddings...")
    corpus_qs = corpus_df['q'].tolist()
    corpus_embs = sbert.encode(corpus_qs, convert_to_tensor=True, show_progress_bar=True, batch_size=128)
    
    # 2. 计算 Query Embedding
    print("   计算 Query Embeddings...")
    q_qs = query_df['q'].tolist()
    q_embs = sbert.encode(q_qs, convert_to_tensor=True, show_progress_bar=True, batch_size=128)
    
    samples = [] # (pos, neg)
    corpus_records = corpus_df.to_dict('records')
    
    eval_data = [] # 用于 MRR 评估: List of {q_id, pos_scores, neg_scores}
    
    for i in tqdm(range(len(query_df)), desc="Building Pairs"):
        q_text = str(q_qs[i])
        gold_ans = str(query_df.iloc[i]['a'])
        q_emb = q_embs[i]
        
        # A. BM25 Retrieve
        scores = bm25.get_scores(q_text.lower().split())
        top_idx = np.argsort(scores)[-args.k_candidates:]
        
        # 准备批量计算 PPL
        candidates = [] # list of dict
        prompts = []
        hashes = []
        
        max_bm25 = max(scores[top_idx]) + 1e-6 # 动态归一化
        
        # 提取候选信息
        cand_indices = top_idx.tolist()
        cand_embs_batch = corpus_embs[cand_indices]
        cos_sims = util.cos_sim(q_emb, cand_embs_batch)[0].cpu().numpy()
        
        temp_cands = []
        
        for k, c_idx in enumerate(top_idx):
            c_row = corpus_records[c_idx]
            c_q = str(c_row['q']).strip()
            # [修复 #2] 在 dev 检索时，如果碰巧检索到自己（虽然理论上 split 了不会，但以防万一），skip
            if normalize_answer(c_q) == normalize_answer(q_text): continue
            
            c_a = str(c_row['a']).strip()
            
            prompt = f"Q: {c_q} A: {c_a}\nQ: {q_text} A:"
            h = get_md5(f"{q_text}_{c_q}_{c_a}")
            
            prompts.append(prompt)
            hashes.append(h)
            
            temp_cands.append({
                'bm25': scores[c_idx] / max_bm25, # Normalized
                'cos': float(cos_sims[k]),
                'ans': c_a
            })
            
        # B. 计算 PPL
        ppl_vals = ppl_engine.get_ppl_batch(prompts, hashes)
        
        # C. 组装特征
        pos_feats = []
        neg_feats = []
        
        for k, cand in enumerate(temp_cands):
            ppl = ppl_vals[k]
            # 特征: [BM25, CosSim, 1/log(PPL)]
            # PPL 越小越好，所以取倒数或者负对数让其“越大越好”
            # log(1) = 0, log(100) = 4.6
            # 使用 1 / (1 + log(PPL))，范围 0-1
            f_ppl = 1.0 / (1.0 + np.log1p(ppl))
            
            feat = [cand['bm25'], cand['cos'], f_ppl]
            
            if is_positive_sample(cand['ans'], gold_ans):
                pos_feats.append(feat)
            else:
                neg_feats.append(feat)
        
        # D. 生成 Pair
        if pos_feats and neg_feats:
            # 训练集生成 Pair
            if "train" in name:
                for pf in pos_feats:
                    negs = random.sample(neg_feats, min(len(neg_feats), args.neg_ratio))
                    for nf in negs:
                        samples.append((pf, nf))
            else:
                # 验证集保留完整列表用于评估 MRR
                # 这里为了简单 Dataset 格式，还是存 Pair，但只存一个代表
                samples.append((pos_feats[0], neg_feats[0])) 
    
    print(f"✅ {name} 生成了 {len(samples)} 对样本")
    torch.save(samples, save_path)
    return samples

# ================= Ranker 模型 =================
class RankerV3(nn.Module):
    def __init__(self):
        super().__init__()
        # 3维特征: [BM25, CosSim, PPL_Score]
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ================= Main =================
def main():
    # 1. 加载全量数据并切分
    full_df = smart_read_csv(args.train_file)
    
    # [修复 #2] 严格切分：最后 10% 做 Dev，剩下做 Train Corpus
    split_idx = int(len(full_df) * 0.9)
    train_corpus = full_df.iloc[:split_idx].reset_index(drop=True) # 检索库
    dev_query_df = full_df.iloc[split_idx:].reset_index(drop=True) # 验证集的 Query
    
    print(f"📊 数据集切分: Train Corpus={len(train_corpus)}, Dev Query={len(dev_query_df)}")
    
    # 2. 准备资源
    print("构建 BM25 (基于 Train Corpus)...")
    bm25 = BM25Okapi([str(t).lower().split() for t in train_corpus['q']])
    
    print(f"加载 SBERT: {EMBEDDING_MODEL_NAME}...")
    sbert = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    print(f"加载 GPT-2 PPL 引擎...")
    ppl_engine = PPLEngine(GPT2_MODEL_NAME, DB_PATH)
    
    # 3. 生成数据 (注意：Dev 也是在 Train Corpus 里搜)
    train_data = build_dataset(train_corpus, train_corpus, bm25, sbert, ppl_engine, "train")
    dev_data = build_dataset(dev_query_df, train_corpus, bm25, sbert, ppl_engine, "dev")
    
    ppl_engine.close()
    
    # 4. 训练
    class MyDS(Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i): 
            return torch.tensor(self.d[i][0], dtype=torch.float), torch.tensor(self.d[i][1], dtype=torch.float)
            
    train_loader = DataLoader(MyDS(train_data), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MyDS(dev_data), batch_size=args.batch_size, shuffle=False)
    
    ranker = RankerV3().to(device)
    optimizer = optim.Adam(ranker.parameters(), lr=args.lr)
    criterion = nn.MarginRankingLoss(margin=0.1)
    
    print("🔥 开始训练 V3 Ranker...")
    for epoch in range(args.epochs):
        ranker.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for pos, neg in pbar:
            pos, neg = pos.to(device), neg.to(device)
            optimizer.zero_grad()
            sp, sn = ranker(pos), ranker(neg)
            loss = criterion(sp, sn, torch.ones_like(sp))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # 简单验证
        ranker.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for pos, neg in val_loader:
                pos, neg = pos.to(device), neg.to(device)
                correct += (ranker(pos) > ranker(neg)).sum().item()
                total += pos.size(0)
        print(f"Epoch {epoch+1} Val Pair Acc: {correct/total:.4f}")
        
    torch.save(ranker.state_dict(), "ranker_v3.pth")
    print("✅ 模型保存为 ranker_v3.pth")

if __name__ == "__main__":
    main()
