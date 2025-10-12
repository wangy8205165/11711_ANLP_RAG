# Team Contributions

**Team Members:** Yixiang Wang, Ibrahim Mohamed Hassan Aldarmaki, Junhao Chen

## Division of Work

### Data Collection & Preprocessing
- **Responsible:** Yixiang Wang, Ibrahim Mohamed Hassan Aldarmaki, Junhao Chen
- **Tasks:**
  - Web scraping and data crawling
  - Data cleaning and formatting
  - Chunk creation and organization
  - Testing questions annotation
- **Description:**
  - Every member handles a subset of the websites and we all goes through the process of scraping, cleaning and annotating of the data on the websites.

### Retrieval System Implementation
- **Responsible:** Yixiang Wang
- **Tasks:**
  - Dense retrieval (FAISS + embeddings)
  - Sparse retrieval (BM25)
  - Hybrid methods (weighted fusion, RRF)
- **Description:**
  - Yixiang mainly implements the RAG and testing of its functionality

### Generation & Evaluation
- **Responsible:** Yixiang Wang, Ibrahim Mohamed Hassan Aldarmaki, Junhao Chen
- **Tasks:**
  - LLM integration (Llama3, DeepSeek)
  - Prompt engineering
  - Testing and evaluation
  - Report & Analysis
- **Description:**
  - Yixiang works on the LLM intergration. Junhao and Ibrahim mainly focus on testing and running experiments. Then, they work on evaluation and anlysis based on the results of the experiments. Ibrahim mainly contributes to the report.


# A summary of each team member's contributions:

## Contributions of each team member

### Junhao
- Processed events, music & culture websites by crawling, cleaning the raw text. 
- Conducted chunking on the raw text. 
- Conducted ablation experiments on different retrieval methods, and topk numbers. 
- Generated the prediction results for Test Set Day 3
- Tested performance of llama3 and deepseek. 

### Ibrahim
- Processed general information websites by crawling, cleaning the raw text. 
- Conducted chunking on the raw text. 
- Built automatic evaluation system that produces score like F1, Cosine Similariy. 
- Developed Judge by LLM method to compute accuracy more efficiently. 
- Drafted and organized the report 

### Yixiang
- Processed sports, food-related information.
- Conducted chunking on the raw text. 
- Build RAG system from scratch, including chunk & query embedding, retrival methods (rrf, weighted,sparse,dense), and final generation using LLM (llama3 and deepseek)
- merged all the chunks, questions, and reference answer. 
- Wrote detailed README file and tutoring the entire team how to run our RAG system. 