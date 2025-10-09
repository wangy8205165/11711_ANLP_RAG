## 11711_ANLP_RAG Homework 2

This is Github repo is for Team (Yixiang Wang, Ibrahim Mohamed Hassan Aldarmaki, Junhao Chen) 


We tested the RAG using two language models: Meta Llama-3.1 Instruct 8B and Deepseek-Distill-QWen-14B 
You can run these two models by running ```generate_llama3.py``` or ```generate_deepseek.py```

## How to run the entire system?
We tested our system on Colab environments, so the following instruction will be tailored only to Colab. 

### Step1-Colab Environment Configurations
1. In your Colab, firstly git clone this repo
2. cd into the directory of this repo
3. ```pip install -r requirements.txt```
4. If you have any issues with environments, manually install following packages:
- ```!pip install faiss-cpu```
- ```!pip install rank-bm25```
- ```!pip install FlagEmbedding==1.2.10```

### Step 2-Running Llama 3 requirements:
1. If you want to run llama-3.1, get a permission on HuggingFace and create a token
2. ```from huggingface_hub import notebook_login```
3. ```notebook_login({Your token})```
4. ```!hf auth whoami``` To Check if you log in correctly. 

### Step 3-Running DeepSeek requirements: 
1. You do not have to do anything for this model. 

### Step 4-Running Embeddings:
1. First step is to run embedding of the target chunk: ```!python embeder.py --chunk {dataset} --model```
2. options for dataset are:
- 
- 
- ...
3. options for embedding models are: 
- 
- 
- 
4. Example: ```!python embeder.py --chunk carnegiemuseum --model transformer-sentence```
5. After running this, the corresponding embeddings ```embeddings_{dataset}.npy``` and ids ```ids_{dataset}.npy``` will be created under ```\index```. 

### Step 5-Running generation:
1. Once you have embeddings and ids, now you can run RAG. 
2. ```!python generate_deepseek.py --mode {retrieval method} --dataset {dataset} --topk {topk}```
3. options for retrieval method:
    - dense
    - sparse
    - weighted
    - rrf
4. options for dataset are the same as above. 
5. topk is the number of top retrieved pieces of information to use. 

### Output
