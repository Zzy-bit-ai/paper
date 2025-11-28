# 在文件最顶部 import 后面加上
import hashlib
import json
from pathlib import Path

# 创建缓存文件夹
Path("cache").mkdir(exist_ok=True)
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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from typing import Optional
from utils import categories, templates, chunks, constructPrompt, cleanLabel, most_common

from dotenv import load_dotenv
from openai import OpenAI



# 加载环境变量
load_dotenv()

# API配置（适配DeepSeek）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', default='generation', type=str,
                   help="classification or generation（问答任务用generation）")
parser.add_argument('--task_name', default='trivia_qa', type=str)
parser.add_argument('--train_name', default='trivia_qa_train_78785_dev_full', type=str)
parser.add_argument('--category_name', default='QA', type=str,
                   help="适配utils中QA模板")
parser.add_argument('--epochs', default=1, type=int,
                   help="通常1-2轮即可")
parser.add_argument('--bz_train', default=3, type=int,
                   help="基础全局示例数量（3-5个）")
parser.add_argument('--knn_num', default=20, type=int,
                   help="KNN候选总数（需大于top_k）")
parser.add_argument('--top_k', default=10, type=int,
                   help="最终保留的Top-k示例（受token限制）")
parser.add_argument('--lambda_ppl', default=0.3, type=float,
                   help="困惑度权重λ（0.0~1.0），语义相似度权重自动为1-λ。λ=0.3表示偏向语义（更像原KATE），λ=0.7表示更看重困惑度")
parser.add_argument('--bz_dev', default=5, type=int)
parser.add_argument('--max_tokens', default=50, type=int)
parser.add_argument('--truncate', action='store_true',
                   help="截断长文本避免超token")
parser.add_argument('--kNN_dev_train', default='trivia_qa_train_78785_dev_full_roberta-large-nli-mean-tokens_cosine_mean', type=str)
parser.add_argument('--PIK_name', default='trivia_qa_kate_ppl', type=str)
parser.add_argument('--compute_perplexity', action='store_true', required=True,
                   help="必须启用，实现KATE + 困惑度加权扩展")
args = parser.parse_args()

##############################################################
# 1. 归一化工具
def normalize(values):
    values = np.array(values)
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return np.full_like(values, 0.5)
    return (values - min_val) / (max_val - min_val)

def normalize_similarity(distances):
    normalized_dist = normalize(distances)
    return 1 - normalized_dist  # 距离越小 → 相似度越高

def normalize_perplexity(ppls):
    ppls = np.array(ppls)
    ppls[np.isinf(ppls)] = np.max(ppls[np.isfinite(ppls)]) * 2
    normalized_ppl = normalize(ppls)
    return 1 - normalized_ppl  # PPL越低 → 分数越高

##############################################################
# 2. 困惑度计算器（批量优化）
class PerplexityCalculator:
    def __init__(self, max_workers=8):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def get_client(self):
        return OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    
    def calculate_perplexity(self, text: str) -> Optional[float]:
        try:
            client = self.get_client()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": text}],
                logprobs=True,
                top_logprobs=1,
                temperature=1.0,   # 建议加temperature避免全是-inf
            )
            logprobs_content = response.choices[0].logprobs.content
            if not logprobs_content:
                return float('inf')
            token_logprobs = [item.logprob for item in logprobs_content if item.logprob is not None]
            if not token_logprobs:
                return float('inf')
            avg_logprob = sum(token_logprobs) / len(token_logprobs)
            nll = -avg_logprob
            return math.exp(nll)
        except Exception as e:
            print(f"PPL计算错误: {e}")
            return float('inf')
    
    def batch_calculate(self, texts: list) -> np.ndarray:
        futures = [self.executor.submit(self.calculate_perplexity, text) for text in texts]
        results = [future.result() for future in as_completed(futures)]
        return np.array(results)
    
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

##############################################################
# 3. 问答生成类
class QAComplete:
    def __init__(self, max_workers=8):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY未配置")
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def __call__(self, example="", max_tokens=50):
        if isinstance(example, list):
            return self._batch_completion(example, max_tokens)
        return [self._single_completion(example, max_tokens)]
    
    def _batch_completion(self, prompts, max_tokens):
        future_to_idx = {self.executor.submit(self._single_completion, p, max_tokens): i for i, p in enumerate(prompts)}
        results = ["" for _ in prompts]
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"生成错误（样本{idx}）: {e}")
        return results
    
    def _single_completion(self, prompt, max_tokens):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                stop=["\n\n"]
            )
            return response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        except Exception as e:
            print(f"生成错误: {e}")
            return ""
    
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

##############################################################
# 4. EM评估工具
PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
ARTICLE_PATTERN = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r'\s+')

def clean_prediction(pred):
    if not isinstance(pred, str):
        pred = str(pred)
    for prefix in ['Answer: ', 'The answer is: ', 'answer: ', 'the answer is: ']:
        if pred.startswith(prefix):
            pred = pred[len(prefix):]
            break
    return pred.strip()

def compute_exact_match(pred, ref):
    def standardize(text, is_pred):
        text = clean_prediction(text) if is_pred else str(text)
        text = PUNCTUATION_PATTERN.sub('', text)
        text = ARTICLE_PATTERN.sub('', text)
        text = WHITESPACE_PATTERN.sub(' ', text).lower().strip()
        return text
    return 1 if standardize(pred, True) == standardize(ref, False) else 0

##############################################################
# 5. 初始化
task = QAComplete()
ppl_calculator = PerplexityCalculator()

# 核心参数
task_name = args.task_name
category_name = args.category_name
epochs = args.epochs
bz_train = args.bz_train
top_k = args.top_k

# 关键：λ 为困惑度权重，语义相似度权重 = 1-λ
lambda_ppl = args.lambda_ppl
alpha_semantic = 1.0 - lambda_ppl   # 语义相似度权重

##############################################################
# 6. 加载数据
try:
    train_fname = os.path.join("dataset", f"{args.train_name}_train.tsv")
    dev_fname = os.path.join("dataset", f"{args.train_name}_dev.tsv")
    
    train_df = pd.read_csv(train_fname, sep='\t', engine='python', header='infer', keep_default_na=False)
    dev_df = pd.read_csv(dev_fname, sep='\t', engine='python', header='infer', keep_default_na=False)
    
    print(f"✅ 加载训练集: {train_fname}（样本数：{len(train_df)}）")
    print(f"✅ 加载验证集: {dev_fname}（样本数：{len(dev_df)}）")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    sys.exit(1)

def detect_qa_columns(df):
    if 'q' in df.columns and 'a' in df.columns:
        return 'q', 'a'
    q_cols = [c for c in df.columns if 'question' in c.lower()]
    a_cols = [c for c in df.columns if 'answer' in c.lower() or 'label' in c.lower()]
    if q_cols and a_cols:
        return q_cols[0], a_cols[0]
    raise ValueError("未找到QA列")

question_col, label_col = detect_qa_columns(train_df)
print(f"✅ 问题列: {question_col}，答案列: {label_col}")

train_labels = train_df[label_col].tolist()
dev_labels_cache = {idx: dev_df.loc[idx, label_col] for idx in range(len(dev_df))}

##############################################################
# 7. 加载KNN（终极防坑版，彻底解决数组真值歧义 + 兼容所有键名）
try:
    knn_fpath = os.path.join("kNN_pretraining", f"{args.kNN_dev_train}.dat")
    
    if not os.path.exists(knn_fpath):
        # 兼容你之前手动指定的绝对路径（Windows）
        knn_fpath = r"D:\py_code\paper\KATEGPT3-main\KATEGPT3-main\inference\kNN_pretraining\trivia_qa_train_78785_dev_full_roberta-large-nli-mean-tokens_cosine_mean.dat"
    
    if not os.path.exists(knn_fpath):
        raise FileNotFoundError(f"KNN文件不存在: {knn_fpath}")
    
    print(f"正在加载KNN文件: {knn_fpath}")
    with open(knn_fpath, "rb") as f:
        knn_data = pickle.load(f)
    
    # 打印键名，方便调试
    print(f"KNN文件包含的键: {list(knn_data.keys())}")
    
    # 安全提取索引（完全避免真值歧义）
    kNN_dev_train = None
    for key in ["kNN_dev_train", "indices", "index", "knn_indices"]:
        if key in knn_data:
            value = knn_data[key]
            if value is not None:
                kNN_dev_train = np.array(value, dtype=np.int64) if not isinstance(value, np.ndarray) else value
                print(f"使用键 '{key}' 作为索引，形状: {kNN_dev_train.shape}")
                break
    
    # 安全提取距离（如果没有就伪造一个，保证代码能跑）
    kNN_distances = None
    for key in ["kNN_distances", "distances", "distance", "knn_distances"]:
        if key in knn_data:
            value = knn_data[key]
            if value is not None:
                kNN_distances = np.array(value, dtype=np.float32) if not isinstance(value, np.ndarray) else value
                print(f"使用键 '{key}' 作为距离，形状: {kNN_distances.shape}")
                break
    
    # 关键：如果没有距离，就伪造一个“从近到远”的距离序列（保证排序逻辑正确）
    if kNN_distances is None:
        print("未找到距离矩阵，使用伪距离（从近到远递增），仅影响语义权重部分")
        pseudo = np.zeros_like(kNN_dev_train, dtype=np.float32)
        for i in range(pseudo.shape[1]):
            pseudo[:, i] = i * 0.02  # 第0个最近，第19个最远
        kNN_distances = pseudo
    
    # 最终检查
    if kNN_dev_train is None:
        raise ValueError("KNN文件中未找到任何有效的索引数据")
    
    print(f"✅ KNN加载成功！")
    print(f"   索引形状: {kNN_dev_train.shape}，距离形状: {kNN_distances.shape}（伪距离: {kNN_distances is pseudo if 'pseudo' in locals() else False}）")

except Exception as e:
    print(f"❌ KNN加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

##############################################################
# 8. 实验跟踪 & 采样
track_train = []
pred_dev = [[]]
accuracy_dev = [[]]
perplexity_records = []

random.seed(30)
dev_sample_num = min(args.bz_dev, len(dev_df))
dev_sample_indices = random.sample(range(len(dev_df)), k=dev_sample_num)
print(f"✅ 采样验证集: {dev_sample_num} 个")

if category_name not in categories:
    Q_list = [f"Question: {{{question_col}}}\nAnswer: "]
    A_list = [f"{{{label_col}}}"]
    templates = {category_name: {"Qs": Q_list, "A": A_list}}
else:
    Q_list = categories[category_name]["Qs"]
    A_list = categories[category_name]["A"]

output_buffer = StringIO()

##############################################################
# 9. 核心实验循环（KATE + PPL 加权）
for epoch in tqdm(range(epochs), desc="实验进度"):
    train_sample_indices = random.sample(range(len(train_df)), k=bz_train)
    track_train.append(train_sample_indices)
    
    base_prompt = constructPrompt(
        df=train_df, labels=train_labels, indices=train_sample_indices,
        templates=templates[category_name],
        Q_list=Q_list, A_list=A_list,
        A_flag=True, truncate=args.truncate
    )
    
    total_em = 0
    batch_size = 5
    for batch in tqdm(chunks(dev_sample_indices, n=batch_size), desc=f"Epoch {epoch+1} 批次", leave=False):
        batch_prompts = []
        batch_refs = []
        batch_ppl_data = []
        
        for dev_idx in batch:
            batch_refs.append(dev_labels_cache[dev_idx])
            
            question_prompt = constructPrompt(
                df=dev_df, labels=[], indices=[dev_idx],
                templates=templates[category_name],
                Q_list=Q_list, A_list=A_list,
                A_flag=False, truncate=args.truncate
            )
            
            # KATE 候选
            candidate_indices = kNN_dev_train[dev_idx][:args.knn_num]
            candidate_distances = kNN_distances[dev_idx][:args.knn_num]
            
            if candidate_indices.size == 0 or candidate_distances.size == 0:
                final_context = base_prompt
                batch_prompts.append(final_context + question_prompt)
                continue
            
            # 语义相似度得分
            similarity_scores = normalize_similarity(candidate_distances)
            
            # 构建候选完整prompt并计算PPL
            candidate_full_prompts = [
                constructPrompt(
                    df=train_df, labels=train_labels, indices=[int(idx)],
                    templates=templates[category_name],
                    Q_list=Q_list, A_list=A_list,
                    A_flag=True, truncate=args.truncate
                ) + question_prompt
                for idx in candidate_indices
            ]
            candidate_ppls = ppl_calculator.batch_calculate(candidate_full_prompts)
            ppl_scores = normalize_perplexity(candidate_ppls)
            
            # 核心：λ 控制困惑度权重，1-λ 控制语义相似度权重
            combined_scores = alpha_semantic * similarity_scores + lambda_ppl * ppl_scores
            
            # 取Top-k
            sorted_idx = np.argsort(combined_scores)[::-1]
            topk_candidate_idx = candidate_indices[sorted_idx[:top_k]].astype(int).tolist()
            
            knn_prompt = constructPrompt(
                df=train_df, labels=train_labels, indices=topk_candidate_idx,
                templates=templates[category_name],
                Q_list=Q_list, A_list=A_list,
                A_flag=True, truncate=args.truncate
            )
            
            final_context = base_prompt + "\n\n" + knn_prompt if knn_prompt else base_prompt
            batch_prompts.append(final_context + question_prompt)
            
            if args.compute_perplexity:
                batch_ppl_data.append({
                    "dev_idx": dev_idx,
                    "candidate_indices": candidate_indices.tolist(),
                    "similarity_scores": similarity_scores.tolist(),
                    "ppl_scores": ppl_scores.tolist(),
                    "combined_scores": combined_scores.tolist(),
                    "topk_indices": topk_candidate_idx,
                    "lambda_ppl": lambda_ppl
                })
        
        # 生成 & 评估
        batch_preds = task(example=batch_prompts, max_tokens=args.max_tokens)
        pred_dev[0].append(batch_preds)
        
        batch_em = [compute_exact_match(p, r) for p, r in zip(batch_preds, batch_refs)]
        total_em += sum(batch_em)
        
        if args.compute_perplexity:
            perplexity_records.extend(batch_ppl_data)
    
    avg_em = total_em / dev_sample_num
    accuracy_dev[0].append(avg_em)
    log_msg = f"Epoch {epoch+1} - λ={lambda_ppl:.2f} - 平均EM: {avg_em:.4f}（样本数：{dev_sample_num}）\n"
    print(log_msg)
    output_buffer.write(log_msg)

##############################################################
# 10. 保存结果
result_dir = os.path.join("result", task_name)
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, f"{args.PIK_name}_lambda{args.lambda_ppl}_results.dat")

result_data = {
    "task": task_name,
    "dataset": args.train_name,
    "config": vars(args),
    "em_scores": accuracy_dev,
    "predictions": pred_dev,
    "tracked_train_indices": track_train,
    "sampled_dev_indices": dev_sample_indices,
    "perplexity_records": perplexity_records if args.compute_perplexity else None,
    "lambda_ppl": args.lambda_ppl
}

with open(result_path, "wb") as f:
    pickle.dump(result_data, f)
print(f"✅ 实验完成！结果已保存：{result_path}")
print(f"   使用困惑度权重 λ = {args.lambda_ppl}（语义权重 = {1-args.lambda_ppl}）")
# import pickle
# with open(r"D:\py_code\paper\KATEGPT3-main\KATEGPT3-main\inference\kNN_pretraining\trivia_qa_train_78785_dev_full_roberta-large-nli-mean-tokens_cosine_mean.dat", "rb") as f:
#     data = pickle.load(f)
#     print(data.keys())