#!/usr/bin/env python3
# finetune_rag.py - True Generalization SFT (Validation-Tracked, Zero Anchors)

import time
import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

from train_kan_llm import KANLanguageModel, TokenizerWrapper, autocast_ctx, resolve_amp_dtype, make_grad_scaler

# ---------------------------------------------------------------------------
# Setup & Paths
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_IN = Path("./checkpoints/kan_model_120m_cosmo_best.pt")
CKPT_OUT = Path("./checkpoints/kan_model_rag_instruct.pt")
TOK_DIR = Path("./data/tokenizer_bpe_32768")

BATCH_SIZE = 8
STEPS = 1500        # ~10 minutes on RTX 2080 Ti to build true copy heads
LR = 4e-5
MIN_LR = 8e-6
MAX_LEN = 512
EVAL_INTERVAL = 100 # Evaluate on unseen validation data every 100 steps

# 1. Load Tokenizer & Base Model
print(f"[init] Using device: {DEVICE}")
print(f"[init] Loading base checkpoint from {CKPT_IN} ...")
rust_tok = ByteLevelBPETokenizer.from_file(str(TOK_DIR / "vocab.json"), str(TOK_DIR / "merges.txt"))
tokenizer = TokenizerWrapper(rust_tok)
eot_id = tokenizer.tok.token_to_id("<|endoftext|>") or 0

ckpt = torch.load(CKPT_IN, map_location=DEVICE, weights_only=False)
m_cfg = ckpt["model_config"]

model = KANLanguageModel(
    vocab_size=m_cfg["vocab_size"],
    dim=m_cfg["dim"],
    num_layers=m_cfg["num_layers"],
    max_len=m_cfg["max_len"],
    k=m_cfg["k"],
    stage_size=m_cfg["stage_size"]
).to(DEVICE)

model.load_state_dict(ckpt["model_state"])

# ---------------------------------------------------------------------------
# 2. Prepare SQuAD Train & Validation Datasets (Zero Anchor Shortcuts)
# ---------------------------------------------------------------------------
print("[data] Loading 'rajpurkar/squad' train and validation splits...")
raw_train = load_dataset("rajpurkar/squad", split="train")
raw_val = load_dataset("rajpurkar/squad", split="validation")

def process_split(ds, limit=10000):
    pairs = []
    for item in ds:
        ctx = item["context"].strip()
        q = item["question"].strip()
        ans = item["answers"]["text"][0].strip() if item["answers"]["text"] else ""
        if not ans or len(ans) > 60:
            continue
        pairs.append(f"Context: {ctx}\nQuestion: {q}\nAnswer: {ans}<|endoftext|>")
        if len(pairs) >= limit:
            break
    return pairs

train_texts = process_split(raw_train, limit=12000)
val_texts = process_split(raw_val, limit=600)

random.seed(42)
random.shuffle(train_texts)
print(f"[data] Train examples: {len(train_texts):,} | Val examples: {len(val_texts):,}")


class QADataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.samples = []
        for t in texts:
            tokens = tokenizer.encode(t)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    max_len = max(len(s) for s in batch)
    x = torch.full((len(batch), max_len - 1), eot_id, dtype=torch.long)
    y = torch.full((len(batch), max_len - 1), -100, dtype=torch.long)
    for i, s in enumerate(batch):
        x[i, :len(s) - 1] = s[:-1]
        y[i, :len(s) - 1] = s[1:]
    return x.to(DEVICE), y.to(DEVICE)


train_loader = DataLoader(QADataset(train_texts, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(QADataset(val_texts, tokenizer, MAX_LEN),
                        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---------------------------------------------------------------------------
# 3. Validation Routine
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_val_loss(model, val_loader, max_batches=25):
    was_training = model.training
    model.eval()
    losses = []
    amp_dtype = resolve_amp_dtype(DEVICE)
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        with autocast_ctx(DEVICE, amp_dtype):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1), ignore_index=-100)
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses) if losses else float("inf")


# ---------------------------------------------------------------------------
# 4. Dense End-to-End Fine-Tuning Loop
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = make_grad_scaler(DEVICE, resolve_amp_dtype(DEVICE))
amp_dtype = resolve_amp_dtype(DEVICE)

model.train()
data_iter = iter(train_loader)
best_val_loss = float("inf")

print("=" * 75)
print(f"True Generalization SFT ({STEPS} steps, validation-guided on RTX 2080 Ti)")
print("=" * 75)

t0 = time.time()
running_train_loss = 0.0

for step in range(1, STEPS + 1):
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x, y = next(data_iter)

    # Cosine learning rate schedule
    progress = step / STEPS
    cur_lr = MIN_LR + (LR - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = cur_lr

    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx(DEVICE, amp_dtype):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1), ignore_index=-100)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    running_loss_item = loss.item()
    running_train_loss += running_loss_item

    # Periodic Validation Check
    if step % EVAL_INTERVAL == 0 or step == STEPS:
        avg_train = running_train_loss / EVAL_INTERVAL
        running_train_loss = 0.0

        val_loss = evaluate_val_loss(model, val_loader, max_batches=25)
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            ckpt["model_state"] = model.state_dict()
            torch.save(ckpt, CKPT_OUT)
            saved_str = "[SAVED NEW BEST]"
        else:
            saved_str = ""

        print(f"Step {step:04d}/{STEPS} | Train: {avg_train:.4f} | Val: {val_loss:.4f} (Best: {best_val_loss:.4f}) {saved_str} | {time.time()-t0:.0f}s")

# Ensure checkpoint was saved at least once
if not CKPT_OUT.exists():
    ckpt["model_state"] = model.state_dict()
    torch.save(ckpt, CKPT_OUT)

print("=" * 75)
print(f"[done] Fine-tuning complete. Best model saved to {CKPT_OUT} with Val Loss: {best_val_loss:.4f}")
print("=" * 75)