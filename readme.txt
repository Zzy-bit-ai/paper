一、QA 生成任务测试（ppl_KATE.py）
1.1 本地 Ollama 模型（qwen:1.8b）- KATE 方法
python ppl_KATE.py
 --task_type generation 
 --category_name QA 
 --task_name trivia_qa_train_78785_dev_full 
 --train_name trivia_qa_train_78785_dev_full 
 --epochs 1 
 --evaluate_dev
 --knn_num 10 
 --max_tokens 10 
 --PIK_name full_roberta-large-nli-mean-tokens_cosine_mean 
 --kNN_dev_train trivia_qa_train_78785_dev_full_roberta-large-nli-mean-tokens_cosine_mean
 --backend ollama
 --ollama_model qwen:1.8b
1.2 在线 DeepSeek 模型（deepseek-chat）- KATE 方法
python ppl_KATE.py
 --task_type generation 
 --category_name QA 
 --task_name trivia_qa_train_78785_dev_full 
 --train_name trivia_qa_train_78785_dev_full 
 --epochs 1 
 --evaluate_dev
 --knn_num 10 
 --max_tokens 10 
 --PIK_name full_roberta-large-nli-mean-tokens_cosine_mean 
 --kNN_dev_train trivia_qa_train_78785_dev_full_roberta-large-nli-mean-tokens_cosine_mean
 --backend deepseek
 --deepseek_model deepseek-chat
二、情感分类任务测试（基于senti_eval.py）
2.1 本地 Ollama 模型（qwen:1.8b）- Baseline 方法
python senti_eval.py
  --task_name sst2 
  --train_name SST-2 
  --dev_name IMDB 
  --epochs 1 
  --eval_size 8 
  --bz_dev 4 
  --k 5 
  --max_tokens 10 
  --mode baseline 
  --backend ollama 
  --ollama_model qwen:1.8b   
2.2 在线 DeepSeek 模型（deepseek-chat）- Baseline 方法
python senti_eval.py
  --task_name sst2 
  --train_name SST-2 
  --dev_name IMDB 
  --epochs 1 
  --eval_size 8
  --bz_dev 4 
  --k 5 
  --max_tokens 10 
  --mode baseline 
  --backend deepseek 
  --deepseek_model deepseek-chat
2.3 本地 Ollama 模型（qwen:1.8b）- KATE 方法
python senti_eval.py
  --task_name sst2 
  --train_name SST-2 
  --dev_name IMDB 
  --epochs 1 
  --eval_size 8 
  --bz_dev 4 
  --k 5 
  --max_tokens 10 
  --mode kate 
  --kNN_dev_train IMDB_roberta-large-nli-mean-tokens_cosine_mean 
  --backend ollama 
  --ollama_model qwen:1.8b    
三、核心参数说明
参数	作用说明
--backend	模型推理后端，ollama对应本地部署模型，deepseek对应在线 API 模型
--mode（情感分类）	测试模式：baseline为随机采样上下文示例，kate为 KNN 检索相似示例
--kNN_dev_train	KATE 方法专用参数，指定预计算的训练集近邻数据文件路径，Baseline 模式无需该参数
--k/--knn_num	上下文示例数量：Baseline 模式为随机采样数，KATE 模式为检索的近邻数量
--PIK_name	结果文件命名，用于区分不同模型 / 方法的测试输出（默认存储于result目录）
--max_tokens	模型生成结果的最大 token 数，避免生成冗余内容
--eval_size	情感分类任务专用，指定验证集总测试样本数
--bz_dev	验证集批次大小，控制单次模型推理的样本数量