import json
import numpy as np
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_chunks(jsonl_path):
    """read chunks.jsonl line by line"""
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data)
    return chunks

def build_embeddings(
        # chunks_path="data/chunks_normal.jsonl",
        chunks_path="data/chunks_littleItaly.jsonl",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        out_dir="index",
        batch_size=128,
        normalize=True):
    
    """Embedding the chunk text using sentence transformer"""
    os.makedirs(out_dir, exist_ok=True)

    # Load the model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Read the chunks
    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]
    ids   = [c["chunk_id"] for c in chunks]

    print(f"Total chunks to encode: {len(texts)}")

    # Encode in Batch
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        emb = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize  # Normalize for cosine similarity
        )
        all_embs.append(emb)

    embs = np.vstack(all_embs).astype("float32")
    ids  = np.array(ids)

    # Save the results
    np.save(os.path.join(out_dir, "embeddings_Italy.npy"), embs)
    np.save(os.path.join(out_dir, "ids_littleItaly.npy"), ids)
    print(f"Saved {embs.shape[0]} embeddings to {out_dir}/")

if __name__ == "__main__":
    build_embeddings(
        chunks_path="data/chunks_littleItaly.jsonl",          # File Path
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        out_dir="index",                     # Output Directory
        batch_size=128,
        normalize=True
    )
