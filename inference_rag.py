#!/usr/bin/env python3
# rag_inference.py
# 120M Decoupled ReLU-KAN + RAG (Context-Biased & Repetition-Constrained)

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from sentence_transformers import SentenceTransformer
import faiss

# Import model architecture and helpers from your training script
from train_kan_llm import KANLanguageModel, TokenizerWrapper, autocast_ctx, resolve_amp_dtype

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------

CKPT_DIR = Path("./checkpoints")
DATA_DIR = Path("./data")

CKPT_INSTRUCT = CKPT_DIR / "kan_model_rag_instruct.pt"
CKPT_BEST = CKPT_DIR / "kan_model_120m_cosmo_best.pt"
CKPT_LATEST = CKPT_DIR / "kan_model_120m_cosmo_latest.pt"
TOK_DIR = DATA_DIR / "tokenizer_bpe_32768"

# ---------------------------------------------------------------------------
# 1. Device & Model Initialization
# ---------------------------------------------------------------------------

def init_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[system] Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Load Tokenizer
    vocab_file = TOK_DIR / "vocab.json"
    merges_file = TOK_DIR / "merges.txt"
    if not vocab_file.exists() or not merges_file.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {TOK_DIR}")

    rust_tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))
    tokenizer = TokenizerWrapper(rust_tok)
    print(f"[tokenizer] Loaded vocab size: {tokenizer.vocab_size:,}")

    # 2. Select Checkpoint (Instruct first, fallback to base)
    if CKPT_INSTRUCT.exists():
        ckpt_path = CKPT_INSTRUCT
        print(f"[model] Loading fine-tuned Instruct checkpoint: {ckpt_path}")
    elif CKPT_BEST.exists():
        ckpt_path = CKPT_BEST
        print(f"[model] Instruct model not found. Falling back to base checkpoint: {ckpt_path}")
    elif CKPT_LATEST.exists():
        ckpt_path = CKPT_LATEST
        print(f"[model] Falling back to latest checkpoint: {ckpt_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_cfg = ckpt.get("model_config", {})

    # 3. Instantiate KAN Model
    model = KANLanguageModel(
        vocab_size=m_cfg.get("vocab_size", tokenizer.vocab_size),
        dim=m_cfg.get("dim", 512),
        num_layers=m_cfg.get("num_layers", 12),
        max_len=m_cfg.get("max_len", 512),
        k=m_cfg.get("k", 4),
        stage_size=m_cfg.get("stage_size", 4)
    ).to(device)

    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    print("[model] 120M ReLU-KAN model loaded successfully into memory.")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# 2. Knowledge Base & Vector Database (FAISS + MiniLM)
# ---------------------------------------------------------------------------

DEFAULT_DOCUMENTS = [
    "Photosynthesis is the biological process used by green plants and algae to synthesize nutrients from carbon dioxide and water using sunlight, generating oxygen as a byproduct.",
    "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. It consists of the Sun and eight planets.",
    "In computer science, an algorithm is a finite sequence of rigorous, step-by-step instructions typically used to solve a specific class of problems or perform computation.",
    "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions, stored in adenosine triphosphate (ATP).",
    "The Treaty of Versailles was the primary peace treaty produced by the Paris Peace Conference at the end of World War I, officially signed on June 28, 1919."
]


class VectorRetriever:
    def __init__(self, documents, device="cpu"):
        print("[retriever] Loading all-MiniLM-L6-v2 embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.documents = list(documents)

        # Build FAISS Index with cosine similarity
        embeddings = self.embedder.encode(self.documents, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"[retriever] FAISS index built with {len(self.documents)} knowledge chunks.")

    def add_document(self, text):
        emb = self.embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(emb.astype(np.float32))
        self.documents.append(text)

    def retrieve(self, query, top_k=1):
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results


# ---------------------------------------------------------------------------
# 3. Constrained & Context-Biased Generation Loop
# ---------------------------------------------------------------------------

@torch.inference_mode()
def answer_rag_query(query, model, tokenizer, retriever, device,
                     context_boost=3.0, max_new_tokens=30):
    # 1. Retrieve the most relevant chunk
    retrieved_chunks = retriever.retrieve(query, top_k=1)
    if not retrieved_chunks:
        return "No relevant context found.", ""
    context = retrieved_chunks[0].strip()

    # 2. Extract unique token IDs present in the context document
    context_tokens = tokenizer.encode(context)
    context_token_set = set(context_tokens)

    # 3. Prompt setup (NO space after Answer:)
    prompt = f"Context: {context}\nQuestion: {query.strip()}\nAnswer:"
    tokens = tokenizer.encode(prompt)

    if len(tokens) > 460:
        truncated_context = context[:len(context) // 2]
        prompt = f"Context: {truncated_context}...\nQuestion: {query.strip()}\nAnswer:"
        tokens = tokenizer.encode(prompt)
        context_token_set = set(tokenizer.encode(truncated_context))

    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    amp_dtype = resolve_amp_dtype(device)
    eot_id = tokenizer.tok.token_to_id("<|endoftext|>")

    generated_tokens = []

    # 4. Constrained Autoregressive Generation
    for step_i in range(max_new_tokens):
        curr_len = input_ids.shape[1]
        if curr_len >= model.max_len:
            break

        idx_cond = input_ids[:, -model.max_len:]
        with autocast_ctx(device, amp_dtype):
            logits = model(idx_cond)

        next_logits = logits[0, -1, :].clone().float()

        # A. Context Logit Boost: prioritize words that actually exist in the retrieved document
        if context_boost > 0.0:
            boost_indices = torch.tensor(list(context_token_set), dtype=torch.long, device=device)
            next_logits[boost_indices] += context_boost

        # B. Strict N-Gram Repetition Filter: mathematically kills "water and water"
        if len(generated_tokens) >= 1:
            # Forbid repeating the immediate previous token
            next_logits[generated_tokens[-1]] = -float("Inf")

        if len(generated_tokens) >= 2:
            # Forbid repeating 2-gram sequences
            last_tok = generated_tokens[-1]
            for j in range(len(generated_tokens) - 1):
                if generated_tokens[j] == last_tok:
                    banned_tok = generated_tokens[j + 1]
                    next_logits[banned_tok] = -float("Inf")

        # C. Greedy Selection
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        tok_str = tokenizer.decode([token_id])

        # D. Clean Stopping Criteria
        if token_id == eot_id or "<|" in tok_str:
            break

        # Stop on newline only if we have already generated at least 2 tokens
        if "\n" in tok_str and len(generated_tokens) >= 2:
            break

        generated_tokens.append(token_id)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    raw_answer = tokenizer.decode(generated_tokens).strip()
    clean_answer = raw_answer.split("<|")[0].strip()
    return clean_answer, context


# ---------------------------------------------------------------------------
# 4. Main Entrypoint & Verification
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("120M Decoupled ReLU-KAN + RAG (Context-Biased & Constrained)")
    print("=" * 70)

    model, tokenizer, device = init_system()
    retriever = VectorRetriever(DEFAULT_DOCUMENTS, device="cpu")

    print("\n--- Running Constrained Verification Checks ---")
    test_queries = [
        "What byproduct is produced during photosynthesis?",
        "When was the Treaty of Versailles signed?",
        "What does an algorithm perform in computer science?",
        "Where is the chemical energy generated by mitochondria stored?",
        "What does the solar system consist of?"
    ]

    for q in test_queries:
        ans, ctx = answer_rag_query(q, model, tokenizer, retriever, device)
        print(f"\n[Question]: {q}")
        print(f"[Retrieved]: {ctx}")
        print(f"[Answer]   : {ans}")

    print("\n" + "=" * 70)
    print("Interactive Mode Ready! Type your question, or 'exit' to quit.")
    print("=" * 70)

    while True:
        try:
            user_query = input("\nQuestion: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ("exit", "quit", "q"):
                print("Exiting RAG session.")
                break

            ans, ctx = answer_rag_query(user_query, model, tokenizer, retriever, device)
            print(f"Answer: {ans}")
            print(f"(Source Context: {ctx})")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()