import json
import torch
import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from retrieve import build_faiss_index, load_chunks, embed_queries, search, load_questions  # import utility function as needed

#  ================== Configuration ==================================
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"   # Choose the model
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_CONTEXT_CHARS = 3000                        # Control the length of context
TOP_K = 3                                       # Many top answers will be used
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------

# Construct Template Prompt
PROMPT_TEMPLATE = """You are a concise and factual assistant.
Answer the question based only on the CONTEXT below.
If the answer cannot be found in the context, say "N/A".
Keep your answer short (within 20 words).

QUESTION: {question}

CONTEXT:
{context}
"""

# def load_questions(path="data/question.txt"):
#     with open(path, "r", encoding="utf-8") as f:
#         return [line.strip() for line in f if line.strip()]
# def build_llm_pipeline(model_id=MODEL_ID, device=DEVICE):
#     print(f"Loading LLM model: {model_id}")
#     tok = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
#     pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")
#     return pipe

def build_llm_pipeline(model_id=MODEL_ID, device=DEVICE):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline


def generate_answer(llm_pipe, question, retrieved_chunks):
    # Concatenate context
    ctxs, total_len = [], 0
    for item in retrieved_chunks:
        t = item["text"].strip()
        if total_len + len(t) > MAX_CONTEXT_CHARS:
            break
        ctxs.append(t)
        total_len += len(t)
    context = "\n\n".join(ctxs)

    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    outputs = llm_pipe(prompt, max_new_tokens=64, do_sample=False) # Call the model to generate output
    # answer = outputs[0]["generated_text"].split("CONTEXT:")[-1].strip()
    # lines = answer.splitlines()
    # final = lines[-1].strip() if lines else answer
    # truncate
    # final = " ".join(final.split()[:20])

    answer = outputs[0]["generated_text"][-1]['content']

    return answer

def main():
    # 1. Load chunk and embedding
    chunk_map = load_chunks("chunks.jsonl")
    index = build_faiss_index("index/embeddings.npy")
    ids = np.load("index/ids.npy", allow_pickle=True)
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 2. load the questions
    questions = load_questions("data/test/question_test.txt")

    # 3. load LLM
    llm = build_llm_pipeline(MODEL_ID, DEVICE)

    # 4. For each question, retrieve, and generate answer based on retrieved info
    results = {}
    for qi, q in enumerate(questions, 1):
        print("="*80)
        print(f"[Q{qi}] {q}")
        # q_emb = embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        # Embed the question and retrieve
        query_embs = embed_queries(embed_model, [q])
        retrieved = search(index, query_embs, ids, chunk_map, top_k=TOP_K)[0]  # 

        # get the answers
        ans = generate_answer(llm, q, retrieved)
        results[str(qi)] = ans

        print(f"→ Answer: {ans}\n")

    # 5. save the results
    with open("system_outputs/system_output_1.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved system_outputs/system_output_llm.json")

if __name__ == "__main__":
    import numpy as np
    main()
