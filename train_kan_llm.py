#!/usr/bin/env python3
# train_kan_llm.py
# Generalized decoupled KAN-LLM trainer.
#
# Throughput-first defaults:
#   * torch.compile supported with clean graph execution (enable with --compile).
#   * periodic eval off (set --eval-interval N to re-enable).
#   * background prefetcher on by default (set --no-prefetch to disable).
#   * no per-log torch.cuda.synchronize(); throughput is windowed + cumulative.
#   * logits fed to F.cross_entropy in autocast dtype.
#   * best.pt atomically stores smoothed EMA weights upon improvement.
#   * checkpoints are written atomically with retries.
#   * Lightweight EMA: tracks active trainable parameters, updates on an
#     8-step compound cadence using batched multi-tensor kernels, with fast in-VRAM swaps.
#   * Unused local heads are pruned from intermediate blocks, freeing >300 MB
#     of VRAM and optimizer state on 4 GB laptop GPUs.
#   * Immediate post-step gradient zeroing minimizes peak stage activation memory.

import os
import sys
import json
import math
import time
import re
import shutil
import argparse
import hashlib
import random
import copy
import queue
import threading
from array import array
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model": {
        "dim": 384,
        "num_layers": 8,
        "max_len": 256,
        "k": 6
    },
    "data": {
        "type": "hf",
        "path": "",
        "hf_name": "HuggingFaceFW/fineweb-edu",
        "hf_config": "sample-10BT",
        "hf_split": "train",
        "hf_revision": "",
        "text_column": "text",
        "data_dir": "./data",
        "tokenizer_vocab_size": 32768,
        "val_fraction": 0.01,
        "max_articles": 130000,
        "append_eot": True,
        "seed": 1337
    },
    "training": {
        "batch_size": 16,
        "steps": 25000,
        "lr": 2.5e-4,
        "min_lr": 2e-5,
        "warmup_steps": 250,
        "weight_decay": 1e-4,
        "stage_size": 4,
        "grad_clip": 1.0,
        "log_interval": 50,
        "eval_interval": 0,
        "checkpoint_interval": 1000,
        "sample_interval": 1000,
        "eval_batches": 20,
        "seed": 1337,
        "compile": False,
        "compile_mode": "default",
        "prefetch": True,
        "prefetch_depth": 2,
        "prefetch_pin": True,
        "ema_decay": 0.9999,
        "ema_interval": 8,
        "ema_device": "cuda"
    },
    "generation": {
        "max_new_tokens": 70,
        "temperature": 0.68,
        "top_k": 40,
        "top_p": 0.90,
        "repetition_penalty": 1.20,
        "prompts": [
            "Photosynthesis is the process by which",
            "The solar system consists of",
            "In computer science, algorithms are"
        ]
    },
    "io": {
        "checkpoint_dir": "./checkpoints",
        "checkpoint_name": "kan_model",
        "config_path": "config.json"
    }
}

# ---------------------------------------------------------------------------
# Config loading / CLI overrides
# ---------------------------------------------------------------------------

def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Generalized Decoupled KAN-LLM trainer")
    p.add_argument("--config", type=str, default=None, help="Path to config.json")
    p.add_argument("--save-config", action="store_true", default=False,
                   help="Save effective configuration (including CLI overrides) back to config.json")
    p.add_argument("--fresh", action="store_true", default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--stage-size", type=int, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--ema-interval", type=int, default=None)
    p.add_argument("--ema-device", type=str, choices=["cuda", "cpu"], default=None)
    p.add_argument("--eval-interval", type=int, default=None,
                   help="Steps between validation evals. 0 = disabled (default).")
    p.add_argument("--eval-batches", type=int, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=None)
    p.add_argument("--sample-interval", type=int, default=None)
    p.add_argument("--data-type", type=str, choices=["text", "hf"], default=None)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--hf-name", type=str, default=None)
    p.add_argument("--hf-config", type=str, default=None)
    p.add_argument("--hf-split", type=str, default=None)
    p.add_argument("--text-column", type=str, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--checkpoint-name", type=str, default=None)
    p.add_argument("--compile", dest="compile", action="store_const", const=True, default=None)
    p.add_argument("--no-compile", dest="compile", action="store_const", const=False)
    p.add_argument("--compile-mode", type=str, default=None,
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--prefetch", dest="prefetch", action="store_const", const=True, default=None)
    p.add_argument("--no-prefetch", dest="prefetch", action="store_const", const=False)
    p.add_argument("--prefetch-pin", dest="prefetch_pin", action="store_const", const=True, default=None)
    return p.parse_args()


def load_config():
    args = parse_args()
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    cfg_path = Path(args.config or DEFAULT_CONFIG["io"]["config_path"])
    if cfg_path.exists():
        with open(cfg_path) as f:
            file_cfg = json.load(f)
        cfg = deep_merge(DEFAULT_CONFIG, file_cfg)
        config_existed = True
    else:
        config_existed = False

    cli_map = {
        "dim": ("model", "dim"), "layers": ("model", "num_layers"),
        "seq_len": ("model", "max_len"),
        "batch_size": ("training", "batch_size"), "steps": ("training", "steps"),
        "lr": ("training", "lr"), "min_lr": ("training", "min_lr"),
        "warmup_steps": ("training", "warmup_steps"),
        "weight_decay": ("training", "weight_decay"),
        "grad_clip": ("training", "grad_clip"),
        "stage_size": ("training", "stage_size"),
        "ema_decay": ("training", "ema_decay"),
        "ema_interval": ("training", "ema_interval"),
        "ema_device": ("training", "ema_device"),
        "eval_interval": ("training", "eval_interval"),
        "eval_batches": ("training", "eval_batches"),
        "checkpoint_interval": ("training", "checkpoint_interval"),
        "sample_interval": ("training", "sample_interval"),
        "data_type": ("data", "type"), "data_path": ("data", "path"),
        "hf_name": ("data", "hf_name"), "hf_config": ("data", "hf_config"),
        "hf_split": ("data", "hf_split"), "text_column": ("data", "text_column"),
        "vocab_size": ("data", "tokenizer_vocab_size"),
        "data_dir": ("data", "data_dir"),
        "checkpoint_dir": ("io", "checkpoint_dir"),
        "checkpoint_name": ("io", "checkpoint_name"),
        "compile": ("training", "compile"),
        "compile_mode": ("training", "compile_mode"),
        "prefetch": ("training", "prefetch"),
        "prefetch_pin": ("training", "prefetch_pin"),
    }
    ns = vars(args)
    for arg_name, (section, key) in cli_map.items():
        v = ns.get(arg_name)
        if v is not None:
            cfg[section][key] = v
    if ns.get("config"):
        cfg["io"]["config_path"] = ns["config"]
    fresh = bool(ns.get("fresh"))
    save_config = bool(ns.get("save_config"))

    validate_config(cfg)
    out_path = Path(cfg["io"]["config_path"])

    if not config_existed:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[config] No config file found. Wrote effective configuration to {out_path}")
    elif save_config:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[config] Automatically saved effective configuration (with CLI switches) to {out_path}")

    return cfg, fresh


def validate_config(cfg):
    m = cfg["model"]
    d = cfg["data"]
    t = cfg["training"]

    if m["dim"] < 32 or m["dim"] % 32 != 0:
        raise ValueError("model.dim must be a positive multiple of 32.")
    if m["num_layers"] < 1:
        raise ValueError("model.num_layers must be >= 1.")
    if m["max_len"] < 1:
        raise ValueError("model.max_len must be >= 1.")
    if m["k"] < 1:
        raise ValueError("model.k must be >= 1.")
    if d["tokenizer_vocab_size"] < 2:
        raise ValueError("data.tokenizer_vocab_size must be >= 2.")
    if not 0.0 <= float(d["val_fraction"]) < 1.0:
        raise ValueError("data.val_fraction must be in [0, 1).")
    if d["type"] not in ("text", "hf"):
        raise ValueError("data.type must be 'text' or 'hf'.")
    if d["type"] == "text" and (not d["path"] or not Path(d["path"]).is_file()):
        raise ValueError(f"data.path must be a valid file when data.type is 'text' (got: {d['path']!r}).")
    if t["batch_size"] < 1 or t["steps"] < 1:
        raise ValueError("training.batch_size and training.steps must be >= 1.")
    if t["stage_size"] < 1:
        raise ValueError("training.stage_size must be >= 1.")
    if t["warmup_steps"] < 0:
        raise ValueError("training.warmup_steps must be >= 0.")
    if t["warmup_steps"] >= t["steps"]:
        print("[config] Warning: warmup_steps >= total steps.")
    if t.get("compile_mode") not in ("default", "reduce-overhead", "max-autotune"):
        raise ValueError("training.compile_mode invalid.")
    if int(t.get("eval_interval", 0)) < 0:
        raise ValueError("training.eval_interval must be >= 0 (0 = disabled).")
    if int(t.get("prefetch_depth", 2)) < 1:
        raise ValueError("training.prefetch_depth must be >= 1.")
    if not 0.0 <= float(t.get("ema_decay", 0.0)) < 1.0:
        raise ValueError("training.ema_decay must be in [0.0, 1.0) (0 disables EMA).")
    if int(t.get("ema_interval", 8)) < 1:
        raise ValueError("training.ema_interval must be >= 1.")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device / AMP helpers
# ---------------------------------------------------------------------------

def resolve_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        for enabler in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
            fn = getattr(torch.backends.cuda, enabler, None)
            if fn is not None:
                try:
                    fn(True)
                except Exception:
                    pass
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_ctx(device, dtype=None):
    if device.type != "cuda" or dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device, amp_dtype):
    enabled = device.type == "cuda" and amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def is_cuda_oom_error(exc):
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ---------------------------------------------------------------------------
# Atomic checkpoint file writer
# ---------------------------------------------------------------------------

def _atomic_torch_save(obj, path, retries=6, initial_backoff=0.25):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    torch.save(obj, tmp)

    try:
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
    except OSError:
        pass

    last_err = None
    delay = initial_backoff
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except OSError as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2

    raise RuntimeError(
        f"Could not atomically replace '{path}' after {retries} attempts "
        f"({type(last_err).__name__}: {last_err}). "
        f"The new checkpoint is preserved at '{tmp}'."
    )


def cleanup_stale_tmp_files(directory):
    try:
        for p in Path(directory).glob("*.pt.tmp"):
            try:
                p.unlink()
                print(f"[checkpoint] Removed stale temp file {p.name}")
            except OSError:
                pass
    except Exception:
        pass


def check_free_disk_space(path, required_mb=3500):
    try:
        free_mb = shutil.disk_usage(str(path)).free / (1024 * 1024)
        if free_mb < required_mb:
            print(f"[checkpoint] WARNING: only {free_mb:,.0f} MB free at {path}; "
                  f"a checkpoint needs roughly {required_mb:,} MB. Saves may fail.")
        else:
            print(f"[checkpoint] Free disk space at {path}: {free_mb:,.0f} MB")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# torch.compile wiring (opt-in)
# ---------------------------------------------------------------------------

def build_compiled_forward_parts(model, batch_size, seq_len, device, compile_mode, stage_size):
    opts = dict(mode=compile_mode, dynamic=False, fullgraph=False)
    cand_blocks = [torch.compile(b, **opts) for b in model.blocks]
    cand_heads = [
        torch.compile(b.local_head, **opts) if not isinstance(b.local_head, nn.Identity) else b.local_head
        for b in model.blocks
    ]

    amp_dtype = resolve_amp_dtype(device)
    dummy_x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    dummy_y = torch.zeros(batch_size * seq_len, dtype=torch.long, device=device)

    num_stages = math.ceil(len(model.blocks) / stage_size)
    h = model.get_initial_embeddings(dummy_x)
    for s in range(num_stages):
        if s > 0:
            h = h.detach()
        start = s * stage_size
        end = min((s + 1) * stage_size, len(model.blocks))
        with autocast_ctx(device, amp_dtype):
            for l in range(start, end):
                h = cand_blocks[l](h)
            logits = cand_heads[end - 1](h)
            warm_loss = F.cross_entropy(logits.view(-1, model.vocab_size), dummy_y)
        warm_loss.backward()

    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return cand_blocks, cand_heads


def maybe_compile_model(model, cfg, device):
    t = cfg["training"]
    eager_blocks = list(model.blocks)
    eager_heads = [b.local_head for b in model.blocks]

    if not t.get("compile", False):
        print("[perf] torch.compile disabled.")
        return eager_blocks, eager_heads
    if device.type != "cuda":
        print("[perf] torch.compile skipped (requires CUDA).")
        return eager_blocks, eager_heads

    compile_mode = t.get("compile_mode") or "default"
    print(f"[perf] Compiling {len(model.blocks)} blocks (mode={compile_mode}) and warming up stages...")
    warm_t0 = time.time()
    try:
        blocks_fwd, heads_fwd = build_compiled_forward_parts(
            model, t["batch_size"], cfg["model"]["max_len"], device, compile_mode, t["stage_size"])
        print(f"[perf] torch.compile ready in {time.time() - warm_t0:.1f}s.")
        return blocks_fwd, heads_fwd
    except Exception as e:
        print(f"[perf] torch.compile failed to warm up ({e}); continuing in eager mode.")
        return eager_blocks, eager_heads


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TokenizerWrapper:
    def __init__(self, rust_tok):
        self.tok = rust_tok
        self.vocab_size = rust_tok.get_vocab_size()

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, tokens):
        return self.tok.decode(list(tokens))


def build_or_load_tokenizer(cfg, documents):
    from tokenizers import ByteLevelBPETokenizer
    d = cfg["data"]
    requested_vocab = int(d["tokenizer_vocab_size"])
    tok_dir = Path(d["data_dir"]) / f"tokenizer_bpe_{requested_vocab}"
    tok_dir.mkdir(parents=True, exist_ok=True)
    vocab_file, merges_file = tok_dir / "vocab.json", tok_dir / "merges.txt"

    if vocab_file.exists() and merges_file.exists():
        rust_tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))
        return TokenizerWrapper(rust_tok)

    print(f"[tokenizer] No trained tokenizer found at {tok_dir}; training with "
          f"requested vocab size {requested_vocab:,}...")
    t0 = time.time()
    rust_tok = ByteLevelBPETokenizer()
    sample = documents if len(documents) <= 200_000 else random.sample(documents, 200_000)
    rust_tok.train_from_iterator(
        sample,
        vocab_size=requested_vocab,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
    )
    rust_tok.save_model(str(tok_dir))
    actual_vocab = rust_tok.get_vocab_size()
    if actual_vocab != requested_vocab:
        print(f"[tokenizer] Note: requested {requested_vocab:,}, trained {actual_vocab:,}.")
    print(f"[tokenizer] Trained in {time.time()-t0:.1f}s -> {tok_dir}")
    return TokenizerWrapper(rust_tok)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_CLEAN_RE = re.compile(r"\n{3,}")


def clean_document(text):
    if not isinstance(text, str):
        return None
    t = _CLEAN_RE.sub("\n\n", text.strip())
    return t if len(t) >= 120 else None


def compute_data_signature(cfg):
    d = cfg["data"]
    if d["type"] == "text":
        p = Path(d["path"])
        if not p.is_file():
            raise FileNotFoundError(f"Data file not found: {p}")
        st = p.stat()
        ident = f"text:{p.resolve()}:{st.st_size}:{int(st.st_mtime)}"
    else:
        ident = f"hf:{d['hf_name']}:{d.get('hf_config') or ''}:{d['hf_split']}:{d.get('hf_revision') or ''}"
    ident += (
        f"|col={d['text_column']}|vocab={d['tokenizer_vocab_size']}"
        f"|eot={d['append_eot']}|max_articles={d.get('max_articles') or 0}"
        f"|val_fraction={d['val_fraction']}|clean=v2"
    )
    return hashlib.sha256(ident.encode()).hexdigest()[:16], ident


def load_documents(cfg):
    d = cfg["data"]
    docs = []
    if d["type"] == "text":
        p = Path(d["path"])
        if not p.is_file():
            raise FileNotFoundError(f"Data file not found: {p}")
        print(f"[data] Reading local text file {p} ...")
        raw = p.read_text(encoding="utf-8", errors="replace")
        for chunk in re.split(r"\n\s*\n", raw):
            t = clean_document(chunk)
            if t:
                docs.append(t)
    elif d["type"] == "hf":
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("The 'datasets' package is required for HF data sources.")
        print(f"[data] Loading HuggingFace dataset {d['hf_name']} (config={d['hf_config'] or 'default'}, "
              f"split={d['hf_split']}) ...")
        kwargs = {}
        if d.get("hf_config"):
            kwargs["name"] = d["hf_config"]
        if d.get("hf_revision"):
            kwargs["revision"] = d["hf_revision"]
        ds = load_dataset(d["hf_name"], split=d["hf_split"], **kwargs)
        limit = int(d.get("max_articles") or 0)
        for i, rec in enumerate(ds):
            if limit and i >= limit:
                break
            t = clean_document(rec.get(d["text_column"]))
            if t:
                docs.append(t)
    else:
        raise ValueError(f"Unknown data.type: {d['type']}")

    if not docs:
        raise RuntimeError("No usable documents were extracted from the data source.")
    print(f"[data] {len(docs):,} documents ready.")
    return docs


def tokenize_and_cache(cfg, documents, tokenizer, signature):
    d = cfg["data"]
    cache_path = Path(d["data_dir"]) / f"tokens_{signature}_v{d['tokenizer_vocab_size']}.pt"

    if cache_path.exists():
        t0 = time.time()
        blob = torch.load(cache_path, weights_only=False)
        if blob.get("data_signature") == signature:
            print(f"[data] Loaded cached tokens from {cache_path} "
                  f"({len(blob['train']):,} train / {len(blob['val']):,} val) in {time.time()-t0:.2f}s")
            return blob["train"], blob["val"]
        print(f"[data] Cache at {cache_path} is stale; re-tokenizing.")

    print(f"[data] Batch-tokenizing {len(documents):,} documents...")
    t0 = time.time()
    eot_id = tokenizer.tok.token_to_id("<|endoftext|>")
    if d["append_eot"] and eot_id is None:
        raise RuntimeError("Tokenizer does not contain the <|endoftext|> special token.")
    eot_id = int(eot_id) if eot_id is not None else -1

    all_tokens = array("i")
    batch_size = 4096
    for start in range(0, len(documents), batch_size):
        chunk = documents[start:start + batch_size]
        encs = tokenizer.tok.encode_batch(chunk)
        for enc in encs:
            all_tokens.extend(enc.ids)
            if d["append_eot"]:
                all_tokens.append(eot_id)
        if (start // batch_size) % 25 == 0:
            print(f"[data]   {start + len(chunk):,}/{len(documents):,} docs | "
                  f"{len(all_tokens):,} tokens")

    tokens = torch.tensor(all_tokens, dtype=torch.int32)
    n_val = int(len(tokens) * float(d["val_fraction"]))
    min_required = cfg["model"]["max_len"] + 1
    n_val = min(n_val, len(tokens) // 10)
    if n_val >= min_required:
        val = tokens[:n_val]
        train = tokens[n_val:]
    else:
        val = tokens
        train = tokens

    if len(train) <= min_required:
        raise RuntimeError(f"Not enough training tokens ({len(train):,} vs required {min_required:,}).")
    Path(d["data_dir"]).mkdir(parents=True, exist_ok=True)
    torch.save({"data_signature": signature, "train": train, "val": val}, cache_path)
    print(f"[data] Tokenized {len(tokens):,} tokens in {time.time()-t0:.1f}s -> {cache_path}")
    return train, val


def load_data(cfg):
    signature, ident = compute_data_signature(cfg)
    print(f"[data] Source signature: {ident}")

    d = cfg["data"]
    cache_path = Path(d["data_dir"]) / f"tokens_{signature}_v{d['tokenizer_vocab_size']}.pt"
    tok_dir = Path(d["data_dir"]) / f"tokenizer_bpe_{d['tokenizer_vocab_size']}"
    vocab_file, merges_file = tok_dir / "vocab.json", tok_dir / "merges.txt"

    if cache_path.exists() and vocab_file.exists() and merges_file.exists():
        t0 = time.time()
        blob = torch.load(cache_path, weights_only=False)
        if blob.get("data_signature") == signature:
            from tokenizers import ByteLevelBPETokenizer
            rust_tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))
            tokenizer = TokenizerWrapper(rust_tok)
            print(f"[data] Loaded cached tokens from {cache_path} "
                  f"({len(blob['train']):,} train / {len(blob['val']):,} val) in {time.time()-t0:.2f}s.")
            return blob["train"], blob["val"], tokenizer, signature
        print(f"[data] Cache at {cache_path} is stale; reloading source data.")

    docs = load_documents(cfg)
    tokenizer = build_or_load_tokenizer(cfg, docs)
    train_tokens, val_tokens = tokenize_and_cache(cfg, docs, tokenizer, signature)
    return train_tokens, val_tokens, tokenizer, signature


def get_batch(data, batch_size, seq_len, device, generator=None):
    """Build one (x, y) batch. Slices in a single transfer to device, halves PCIe traffic
    by transferring as int32, and converts to int64 on GPU in <1 us."""
    window_count = len(data) - seq_len
    if window_count <= 0:
        raise ValueError(
            f"Dataset has {len(data):,} tokens, but seq_len={seq_len} requires "
            f"at least {seq_len + 1:,} tokens."
        )
    starts = torch.randint(window_count, (batch_size,), generator=generator)
    seq = data.unfold(0, seq_len + 1, 1).index_select(0, starts)
    if device.type == "cuda":
        seq_dev = seq.to(device, non_blocking=True).long()
    else:
        seq_dev = seq.long()
    return seq_dev[:, :-1], seq_dev[:, 1:]


# ---------------------------------------------------------------------------
# Optional prefetcher
# ---------------------------------------------------------------------------

class PrefetchLoader:
    def __init__(self, data, batch_size, seq_len, device, seed=None,
                 depth=2, pin=False):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.pin = pin
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)
        self._gen_lock = threading.Lock()

        self._q = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._err = None
        self._t = None
        self._started = False
        self._start_lock = threading.Lock()

    def _make(self):
        window_count = len(self.data) - self.seq_len
        if window_count <= 0:
            raise ValueError(
                f"Dataset has {len(self.data):,} tokens, but seq_len={self.seq_len} "
                f"requires at least {self.seq_len + 1:,} tokens."
            )
        with self._gen_lock:
            starts = torch.randint(window_count, (self.batch_size,), generator=self.gen)
        seq = self.data.unfold(0, self.seq_len + 1, 1).index_select(0, starts)
        if self.pin:
            seq = seq.pin_memory()
        return seq

    def _worker(self):
        try:
            while not self._stop.is_set():
                try:
                    self._q.put(self._make(), timeout=0.5)
                except queue.Full:
                    continue
        except Exception as e:
            self._err = e
            try:
                self._q.put(None, timeout=0.5)
            except queue.Full:
                pass

    def _ensure_started(self):
        if self._started:
            return
        with self._start_lock:
            if self._started:
                return
            self._started = True
            self._t = threading.Thread(target=self._worker, daemon=True)
            self._t.start()

    def get(self):
        self._ensure_started()
        item = self._q.get()
        if item is None:
            if self._err is not None:
                raise self._err
            raise RuntimeError("PrefetchLoader queue was closed or returned None.")
        if self.device.type == "cuda":
            seq_dev = item.to(self.device, non_blocking=self.pin).long()
        else:
            seq_dev = item.long()
        return seq_dev[:, :-1], seq_dev[:, 1:]

    def get_state(self):
        with self._gen_lock:
            return self.gen.get_state()

    def set_state(self, state):
        with self._gen_lock:
            if isinstance(state, torch.Tensor):
                state = state.detach().to(device="cpu", dtype=torch.uint8)
            self.gen.set_state(state)

    def close(self):
        if not self._started:
            return
        self._stop.set()
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self._t is not None:
            self._t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self._build_tables(max_len)

    def _build_tables(self, max_len, device=None, dtype=torch.float32):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        t = torch.arange(max_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos2 = torch.cat([cos, cos], dim=-1).view(1, 1, max_len, self.dim).to(dtype=dtype)
        sin2 = torch.cat([sin, sin], dim=-1).view(1, 1, max_len, self.dim).to(dtype=dtype)
        self.register_buffer("cos2", cos2, persistent=False)
        self.register_buffer("sin2", sin2, persistent=False)

    def forward(self, q, k):
        T = q.shape[2]
        if T > self.cos2.shape[2]:
            self._build_tables(max(T, self.cos2.shape[2] * 2), device=q.device, dtype=q.dtype)
        elif self.cos2.device != q.device or self.cos2.dtype != q.dtype:
            self.cos2 = self.cos2.to(device=q.device, dtype=q.dtype)
            self.sin2 = self.sin2.to(device=q.device, dtype=q.dtype)

        cos = self.cos2[:, :, :T]
        sin = self.sin2[:, :, :T]
        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot


class FlashRoPECausalAttention(nn.Module):
    def __init__(self, dim, n_heads=12, max_len=1024):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_len=max_len)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class ReLUKANLinear(nn.Module):
    def __init__(self, in_features, out_features, k=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.base_linear = nn.Linear(in_features, out_features, bias=False)
        grid = torch.linspace(-1.5, 1.5, k)
        self.register_buffer("grid", grid.view(1, 1, 1, k))
        std = 1.0 / math.sqrt(in_features * k)
        self.kan_weight = nn.Parameter(torch.randn(out_features, in_features * k) * std)

    def forward(self, x):
        base_out = self.base_linear(F.silu(x))
        grid = self.grid.to(dtype=x.dtype) if self.grid.dtype != x.dtype else self.grid
        x_norm = torch.tanh(x).unsqueeze(-1)
        basis = torch.relu(x_norm - grid).square()
        kan_out = F.linear(basis.flatten(2), self.kan_weight)
        return base_out + kan_out


class GatedKANFeedForward(nn.Module):
    def __init__(self, dim, k=4):
        super().__init__()
        self.gate_kan = ReLUKANLinear(dim, dim * 2, k=k)
        self.up_linear = nn.Linear(dim, dim * 2, bias=False)
        self.down_linear = nn.Linear(dim * 2, dim, bias=False)
        self.post_norm = nn.LayerNorm(dim * 2)

    def forward(self, x):
        gated = self.gate_kan(x) * F.silu(self.up_linear(x))
        return self.down_linear(self.post_norm(gated))


class FastDecoupledBlock(nn.Module):
    def __init__(self, dim, vocab_size, n_heads=12, k=4, total_layers=8, max_len=1024, has_head=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = FlashRoPECausalAttention(dim, n_heads=n_heads, max_len=max_len)
        self.ln2 = nn.LayerNorm(dim)
        self.kan_ffn = GatedKANFeedForward(dim, k=k)
        self.res_scale = 1.0 / math.sqrt(2.0 * total_layers)

        # Only stage boundary blocks allocate local heads. Intermediate blocks save >50 MB VRAM each.
        if has_head:
            self.local_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim, bias=False),
                nn.SiLU(),
                nn.Linear(dim, vocab_size, bias=False)
            )
        else:
            self.local_head = nn.Identity()

    def forward(self, x):
        x = torch.add(x, self.attn(self.ln1(x)), alpha=self.res_scale)
        x = torch.add(x, self.kan_ffn(self.ln2(x)), alpha=self.res_scale)
        return x

    def compute_local_logits(self, x):
        return self.local_head(x)


class KANLanguageModel(nn.Module):
    def __init__(self, vocab_size=32768, dim=384, num_layers=8, max_len=256, k=4, stage_size=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, dim)
        n_heads = dim // 32

        num_stages = math.ceil(num_layers / stage_size)
        active_head_indices = {min((s + 1) * stage_size, num_layers) - 1 for s in range(num_stages)}

        self.blocks = nn.ModuleList([
            FastDecoupledBlock(dim=dim, vocab_size=vocab_size, n_heads=n_heads, k=k,
                               total_layers=num_layers, max_len=max_len,
                               has_head=(i in active_head_indices))
            for i in range(num_layers)
        ])

    def get_initial_embeddings(self, idx):
        return self.tok_emb(idx)

    def forward(self, idx):
        h = self.get_initial_embeddings(idx)
        for block in self.blocks:
            h = block(h)
        return self.blocks[-1].compute_local_logits(h)


KANLanguageModel32k = KANLanguageModel


# ---------------------------------------------------------------------------
# Lightweight, VRAM-Safe EMA
# ---------------------------------------------------------------------------

class LightweightEMA:
    def __init__(self, active_named_params, decay=0.9999, update_interval=8, device="cuda"):
        self.decay = float(decay)
        self.update_interval = max(1, int(update_interval))
        self.effective_decay = self.decay ** self.update_interval
        self._step_count = 0
        self.target_device = torch.device(device)

        self._names = [name for name, _ in active_named_params]
        self._param_list = [p for _, p in active_named_params]
        self._name_to_idx = {name: i for i, name in enumerate(self._names)}

        # Shadow in float32 for precision
        self.shadow = [
            p.detach().to(device=self.target_device, dtype=torch.float32).clone()
            for p in self._param_list
        ]

    @torch.no_grad()
    def update(self, step=None):
        if step is not None:
            self._step_count = step
        else:
            self._step_count += 1
        if self._step_count % self.update_interval != 0:
            return

        d = self.effective_decay
        one_minus_d = 1.0 - d

        if (self.target_device.type == "cuda"
                and len(self._param_list) > 0
                and self._param_list[0].device.type == "cuda"
                and hasattr(torch, "_foreach_mul_")
                and hasattr(torch, "_foreach_add_")):
            torch._foreach_mul_(self.shadow, d)
            torch._foreach_add_(self.shadow, self._param_list, alpha=one_minus_d)
        else:
            for s, p in zip(self.shadow, self._param_list):
                p_val = p.detach().to(device=self.target_device, dtype=torch.float32)
                s.mul_(d).add_(p_val, alpha=one_minus_d)

    @torch.no_grad()
    def reset_from(self, model=None):
        for s, p in zip(self.shadow, self._param_list):
            s.copy_(p.detach().to(device=self.target_device, dtype=torch.float32))

    @torch.no_grad()
    def copy_to(self, model=None):
        if (hasattr(torch, "_foreach_copy_")
                and len(self._param_list) > 0
                and self._param_list[0].device == self.target_device):
            torch._foreach_copy_(self._param_list, [s.to(dtype=p.dtype) for s, p in zip(self.shadow, self._param_list)])
        else:
            for p, s in zip(self._param_list, self.shadow):
                p.copy_(s.to(device=p.device, dtype=p.dtype))

    def state_dict(self):
        # Save in bfloat16 if hardware supports it, else float16 to prevent CUDA invalid kernel errors on Turing/Pascal GPUs
        save_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        return {
            "decay": self.decay,
            "update_interval": self.update_interval,
            "step_count": self._step_count,
            "shadow": {name: s.to(save_dtype) for name, s in zip(self._names, self.shadow)}
        }

    def load_state_dict(self, sd):
        if not sd:
            return
        self.decay = float(sd.get("decay", self.decay))
        self.effective_decay = self.decay ** self.update_interval
        self._step_count = int(sd.get("step_count", self._step_count))
        shadow_dict = sd.get("shadow", {})
        for name, idx in self._name_to_idx.items():
            t = shadow_dict.get(name)
            if t is not None and t.shape == self.shadow[idx].shape:
                self.shadow[idx].copy_(t.to(device=self.target_device, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MacroDecoupledTrainer32k:
    def __init__(self, model, lr=2e-4, min_lr=2e-5, warmup_steps=100,
                 weight_decay=1e-4, total_steps=25000, stage_size=4,
                 grad_clip=1.0, device=None, blocks_fwd=None, heads_fwd=None,
                 prefetcher=None, ema_decay=0.0, ema_interval=8, ema_device="cuda"):
        self.model = model
        self.device = device or torch.device("cpu")
        self.total_steps = total_steps
        self.base_lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.stage_size = stage_size
        self.grad_clip = grad_clip
        self.num_stages = math.ceil(len(model.blocks) / stage_size)
        self.prefetcher = prefetcher

        self.blocks_fwd = blocks_fwd if blocks_fwd is not None else list(model.blocks)
        self.heads_fwd = heads_fwd if heads_fwd is not None else [b.local_head for b in model.blocks]

        self._stage_bounds = []
        for s in range(self.num_stages):
            start = s * stage_size
            end = min((s + 1) * stage_size, len(model.blocks))
            self._stage_bounds.append((start, end))

        self.optimizers = []
        self._stage_param_lists = []
        self._stage_active_params = []
        active_named_params = []

        for s in range(self.num_stages):
            start, end = self._stage_bounds[s]
            params = []
            if s == 0:
                params.extend(model.tok_emb.parameters())
            for l in range(start, end):
                params.extend(model.blocks[l].parameters())
            self._stage_param_lists.append(params)

            active = []
            if s == 0:
                for name, p in model.tok_emb.named_parameters():
                    active.append(p)
                    active_named_params.append((f"tok_emb.{name}", p))
            for l in range(start, end):
                for m_name in ("ln1", "attn", "ln2", "kan_ffn"):
                    subm = getattr(model.blocks[l], m_name)
                    for name, p in subm.named_parameters():
                        active.append(p)
                        active_named_params.append((f"blocks.{l}.{m_name}.{name}", p))
            for name, p in model.blocks[end - 1].local_head.named_parameters():
                active.append(p)
                active_named_params.append((f"blocks.{end - 1}.local_head.{name}", p))

            self._stage_active_params.append(active)

            opt_kwargs = dict(lr=lr, weight_decay=weight_decay)
            if self.device.type == "cuda":
                try:
                    self.optimizers.append(torch.optim.AdamW(params, fused=True, **opt_kwargs))
                    continue
                except (TypeError, RuntimeError):
                    pass
            self.optimizers.append(torch.optim.AdamW(params, **opt_kwargs))

        self.amp_dtype = resolve_amp_dtype(self.device)
        self.scaler = make_grad_scaler(self.device, self.amp_dtype)

        if ema_decay and ema_decay > 0.0:
            chosen_dev = ema_device if (ema_device == "cpu" or self.device.type != "cuda") else "cuda"
            self.ema = LightweightEMA(
                active_named_params, decay=ema_decay,
                update_interval=ema_interval, device=chosen_dev
            )
            print(f"[ema] Enabled (decay={ema_decay}, interval={ema_interval}, device={chosen_dev}). "
                  f"Active params tracked: {len(active_named_params):,} (~{sum(p.numel() for _, p in active_named_params)*4/(1024**2):.1f} MB).")
        else:
            self.ema = None
            print("[ema] Disabled.")

    def update_learning_rates(self, step):
        if step < self.warmup_steps:
            cur_lr = self.base_lr * (step / float(self.warmup_steps)) if self.warmup_steps > 0 else self.base_lr
        else:
            progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            progress = min(1.0, max(0.0, progress))
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            cur_lr = max(self.min_lr + (self.base_lr - self.min_lr) * decay, self.min_lr)

        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
        return cur_lr

    def _zero_all_grads(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    @contextmanager
    def ema_weights(self):
        if self.ema is None:
            yield
            return
        use_cuda_backup = (self.device.type == "cuda" and self.ema.target_device.type == "cuda")
        if use_cuda_backup:
            backup = [p.detach().clone() for p in self.ema._param_list]
        else:
            backup = [p.detach().to("cpu", copy=True) for p in self.ema._param_list]
        with torch.no_grad():
            self.ema.copy_to()
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(self.ema._param_list, backup):
                    p.copy_(b)

    def train_step(self, x, y, step, should_log=False):
        cur_lr = self.update_learning_rates(step)
        vocab_size = self.model.vocab_size
        blocks_fwd = self.blocks_fwd
        heads_fwd = self.heads_fwd
        bounds = self._stage_bounds
        optimizer_list = self.optimizers
        stage_active_params = self._stage_active_params

        y_flat = y.reshape(-1)
        h = self.model.get_initial_embeddings(x)
        stage_losses = []
        scaler_enabled = self.scaler.is_enabled()
        scaler = self.scaler

        for s in range(self.num_stages):
            if s > 0:
                h = h.detach()

            start, end = bounds[s]

            with autocast_ctx(self.device, self.amp_dtype):
                for l in range(start, end):
                    h = blocks_fwd[l](h)
                logits = heads_fwd[end - 1](h)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y_flat)
                del logits

            optimizer_list[s].zero_grad(set_to_none=True)
            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_list[s])
                if self.grad_clip > 0:
                    try:
                        torch.nn.utils.clip_grad_norm_(stage_active_params[s], max_norm=self.grad_clip, foreach=True)
                    except Exception:
                        torch.nn.utils.clip_grad_norm_(stage_active_params[s], max_norm=self.grad_clip)
                scaler.step(optimizer_list[s])
            else:
                loss.backward()
                if self.grad_clip > 0:
                    try:
                        torch.nn.utils.clip_grad_norm_(stage_active_params[s], max_norm=self.grad_clip, foreach=True)
                    except Exception:
                        torch.nn.utils.clip_grad_norm_(stage_active_params[s], max_norm=self.grad_clip)
                optimizer_list[s].step()

            # Free stage gradients immediately to maximize free VRAM for subsequent stages
            optimizer_list[s].zero_grad(set_to_none=True)

            if should_log:
                stage_losses.append(loss.detach())

        if scaler_enabled:
            scaler.update()

        if self.ema is not None:
            self.ema.update(step)

        if should_log:
            losses = torch.stack(stage_losses).tolist()
            final_loss = losses[-1]
            avg_loss = sum(losses) / len(losses)
            return final_loss, avg_loss, cur_lr
        return 0.0, 0.0, cur_lr

    def save_checkpoint(self, path, step, best_loss, best_val_loss, data_signature,
                        tokens_seen, model_config):
        checkpoint = {
            "step": step,
            "best_loss": best_loss,
            "best_val_loss": best_val_loss,
            "data_signature": data_signature,
            "tokens_seen": tokens_seen,
            "model_config": model_config,
            "model_state": self.model.state_dict(),
            "stage_optimizer_states": [opt.state_dict() for opt in self.optimizers],
            "rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "scaler_state": self.scaler.state_dict(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.ema is not None:
            checkpoint["ema_state"] = self.ema.state_dict()
        if self.prefetcher is not None:
            try:
                checkpoint["prefetcher_rng_state"] = self.prefetcher.get_state()
            except Exception:
                pass
        if self.device.type == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        _atomic_torch_save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state, strict=False)

        saved_opts = checkpoint.get("stage_optimizer_states") or []
        if len(saved_opts) == len(self.optimizers):
            for opt, st in zip(self.optimizers, saved_opts):
                try:
                    opt.load_state_dict(st)
                except Exception as e:
                    print(f"[resume] Optimizer state load skipped ({e}).")
        else:
            print(f"[resume] Optimizer count mismatch ({len(saved_opts)} vs {len(self.optimizers)}).")

        # In PyTorch, torch.cuda.set_rng_state_all expects CPU ByteTensors.
        if checkpoint.get("cuda_rng_state") is not None and self.device.type == "cuda":
            try:
                states = []
                for s in checkpoint["cuda_rng_state"]:
                    if isinstance(s, torch.Tensor):
                        s = s.detach().to(device="cpu", dtype=torch.uint8)
                    states.append(s)
                torch.cuda.set_rng_state_all(states)
            except Exception as e:
                print(f"[resume] Could not restore CUDA RNG state: {e}")

        if checkpoint.get("rng_state") is not None:
            rng_state = checkpoint["rng_state"]
            if isinstance(rng_state, torch.Tensor):
                rng_state = rng_state.detach().to(device="cpu", dtype=torch.uint8)
            try:
                torch.set_rng_state(rng_state)
            except Exception as e:
                print(f"[resume] Could not restore torch RNG state: {e}")

        if checkpoint.get("python_rng_state") is not None:
            try:
                random.setstate(checkpoint["python_rng_state"])
            except Exception:
                pass

        if self.prefetcher is not None and checkpoint.get("prefetcher_rng_state") is not None:
            try:
                pf_state = checkpoint["prefetcher_rng_state"]
                if isinstance(pf_state, torch.Tensor):
                    pf_state = pf_state.detach().to(device="cpu", dtype=torch.uint8)
                self.prefetcher.set_state(pf_state)
            except Exception:
                pass

        scaler_state = checkpoint.get("scaler_state")
        if scaler_state:
            try:
                self.scaler.load_state_dict(scaler_state)
            except Exception:
                pass

        if self.ema is not None:
            ema_state = checkpoint.get("ema_state")
            if ema_state:
                try:
                    self.ema.load_state_dict(ema_state)
                    print(f"[ema] Restored EMA state (decay={self.ema.decay}).")
                except Exception as e:
                    print(f"[ema] Could not restore EMA state: {e}; initializing from loaded model.")
                    self.ema.reset_from()
            else:
                print("[ema] No EMA state in checkpoint; initializing from loaded weights.")
                self.ema.reset_from()

        summary = {
            "step": checkpoint.get("step", 0),
            "best_loss": checkpoint.get("best_loss", float("inf")),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "data_signature": checkpoint.get("data_signature", ""),
            "tokens_seen": checkpoint.get("tokens_seen", 0),
            "model_config": checkpoint.get("model_config", {}),
            "saved_at": checkpoint.get("saved_at", ""),
            "has_ema": "ema_state" in checkpoint,
        }
        del checkpoint
        return summary


# ---------------------------------------------------------------------------
# Evaluation / generation
# ---------------------------------------------------------------------------

def perplexity(loss):
    if math.isnan(loss) or loss == float("inf"):
        return float("inf")
    return math.exp(min(loss, 20.0))


@torch.inference_mode()
def evaluate(model, val_tokens, cfg, device, blocks_fwd=None, heads_fwd=None):
    t = cfg["training"]
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(t["seed"])
    amp_dtype = resolve_amp_dtype(device)
    blocks_fwd = blocks_fwd if blocks_fwd is not None else list(model.blocks)
    heads_fwd = heads_fwd if heads_fwd is not None else [b.local_head for b in model.blocks]
    losses = []
    try:
        for _ in range(int(t["eval_batches"])):
            x, y = get_batch(val_tokens, t["batch_size"], cfg["model"]["max_len"], device, g)
            with autocast_ctx(device, amp_dtype):
                h = model.get_initial_embeddings(x)
                for cb in blocks_fwd:
                    h = cb(h)
                logits = heads_fwd[-1](h)
                loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.reshape(-1))
            losses.append(loss)
    finally:
        model.train(was_training)
    if not losses:
        return float("inf")
    return torch.stack(losses).mean().item()


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p,
             repetition_penalty, device):
    was_training = model.training
    model.eval()
    amp_dtype = resolve_amp_dtype(device)
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    eot_id = tokenizer.tok.token_to_id("<|endoftext|>")

    try:
        for _ in range(max_new_tokens):
            curr_len = input_ids.shape[1]
            slice_len = min(curr_len, model.max_len)
            idx_cond = input_ids[:, curr_len - slice_len:curr_len]

            with autocast_ctx(device, amp_dtype):
                logits = model(idx_cond)
            next_logits = logits[:, -1, :].clone().float()

            recent_len = min(curr_len, 20)
            recent = set(input_ids[0, curr_len - recent_len:curr_len].tolist())
            for tok in recent:
                if next_logits[0, tok] > 0:
                    next_logits[0, tok] /= repetition_penalty
                else:
                    next_logits[0, tok] *= repetition_penalty

            next_logits = next_logits / max(temperature, 1e-4)

            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("Inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                next_logits[0, sorted_indices[sorted_indices_to_remove]] = -float("Inf")

            probs = F.softmax(next_logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() == 0:
                break
            next_token = torch.multinomial(probs, num_samples=1)
            if eot_id is not None and next_token.item() == eot_id:
                break
            input_ids = torch.cat([input_ids, next_token], dim=1)

            new_tok_str = tokenizer.decode([next_token.item()])
            if "." in new_tok_str:
                curr_text = tokenizer.decode(input_ids[0, len(tokens):].tolist())
                if curr_text.count(".") >= 2 and curr_text.strip().endswith("."):
                    break
    finally:
        model.train(was_training)

    return tokenizer.decode(input_ids[0].tolist()).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg, fresh = load_config()
    set_seed(cfg["training"]["seed"])

    device = resolve_device()
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    ckpt_dir = Path(cfg["io"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["io"]["checkpoint_name"]
    latest_path = ckpt_dir / f"{name}_latest.pt"
    best_path = ckpt_dir / f"{name}_best.pt"
    log_path = ckpt_dir / f"{name}_log.jsonl"

    t = cfg["training"]
    eval_interval = int(t.get("eval_interval", 0) or 0)
    ema_decay = float(t.get("ema_decay", 0.0) or 0.0)
    ema_interval = int(t.get("ema_interval", 8) or 8)
    ema_device = str(t.get("ema_device", "cuda") or "cuda")
    checkpoint_interval = int(t.get("checkpoint_interval", 1000) or 1000)
    sample_interval = int(t.get("sample_interval", 1000) or 1000)

    print("=" * 70)
    print(f"Hardware: {device} ({gpu_name})")
    print(f"Data: {cfg['data']['type']} | Requested vocab: {cfg['data']['tokenizer_vocab_size']:,} | "
          f"Context: {cfg['model']['max_len']} | Batch: {t['batch_size']}")
    print(f"Checkpoints: {latest_path} (+ {best_path.name})")
    print(f"Throughput settings: compile={t.get('compile', False)} | "
          f"eval_interval={eval_interval} | checkpoint_interval={checkpoint_interval} | "
          f"ema_decay={ema_decay} (interval={ema_interval}, device={ema_device})")
    print("=" * 70)

    cleanup_stale_tmp_files(ckpt_dir)
    check_free_disk_space(ckpt_dir, required_mb=3500)

    train_tokens, val_tokens, tokenizer, data_signature = load_data(cfg)
    print(f"[tokenizer] Effective vocabulary size: {tokenizer.vocab_size:,}")

    # Pin memory for zero-copy DMA host-to-device transfers
    if device.type == "cuda":
        try:
            if not train_tokens.is_pinned():
                train_tokens = train_tokens.pin_memory()
            if not val_tokens.is_pinned():
                val_tokens = val_tokens.pin_memory()
            print("[perf] Dataset buffers pinned in host RAM for zero-copy transfers.")
        except Exception:
            pass

    model = KANLanguageModel(
        vocab_size=tokenizer.vocab_size,
        dim=cfg["model"]["dim"],
        num_layers=cfg["model"]["num_layers"],
        max_len=cfg["model"]["max_len"],
        k=cfg["model"]["k"],
        stage_size=t["stage_size"]
    ).to(device)

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "dim": cfg["model"]["dim"],
        "num_layers": cfg["model"]["num_layers"],
        "max_len": cfg["model"]["max_len"],
        "k": cfg["model"]["k"],
        "stage_size": t["stage_size"],
    }

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} (~{total_params * 4 / (1024**2):.2f} MB fp32)")

    blocks_fwd, heads_fwd = maybe_compile_model(model, cfg, device)

    prefetcher = None
    if t.get("prefetch", False):
        print(f"[perf] Prefetcher enabled (depth={t.get('prefetch_depth', 2)}, "
              f"pin={t.get('prefetch_pin', False)}).")
        prefetcher = PrefetchLoader(
            train_tokens, t["batch_size"], cfg["model"]["max_len"], device,
            seed=t["seed"], depth=int(t.get("prefetch_depth", 2)),
            pin=bool(t.get("prefetch_pin", False)))

    trainer = MacroDecoupledTrainer32k(
        model, lr=t["lr"], min_lr=t["min_lr"], warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"], total_steps=t["steps"],
        stage_size=t["stage_size"], grad_clip=t["grad_clip"], device=device,
        blocks_fwd=blocks_fwd, heads_fwd=heads_fwd, prefetcher=prefetcher,
        ema_decay=ema_decay, ema_interval=ema_interval, ema_device=ema_device)
    model.train()

    start_step, best_loss, best_val_loss, tokens_seen = 1, float("inf"), float("inf"), 0
    last_saved_checkpoint_step = -1

    try:
        if not fresh and latest_path.exists():
            print(f"\n[resume] Loading {latest_path} ...")
            ckpt = trainer.load_checkpoint(latest_path)
            saved_step = ckpt.get("step", 0)
            ckpt_sig = ckpt.get("data_signature", "")
            ckpt_cfg = ckpt.get("model_config", {})

            mismatch = False
            for k_cfg, v_cfg in model_config.items():
                if k_cfg in ckpt_cfg and ckpt_cfg[k_cfg] != v_cfg:
                    mismatch = True
                    break
            if mismatch:
                raise SystemExit(
                    f"[config-mismatch] Checkpoint was trained with model_config={ckpt_cfg}, "
                    f"but current config gives {model_config}. Align config.json (or use --fresh).")

            start_step = saved_step + 1
            tokens_seen = ckpt.get("tokens_seen", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))

            if ckpt_sig and ckpt_sig != data_signature:
                print(f"[data] Source changed since checkpoint ({ckpt_sig} -> {data_signature}).")
                print("[data] Resetting best-loss tracking; previous values are not comparable.")
                best_loss, best_val_loss = float("inf"), float("inf")
                if best_path.exists():
                    stale = best_path.with_suffix(".stale.pt")
                    try:
                        best_path.replace(stale)
                        print(f"[data] Old best checkpoint moved to {stale}")
                    except OSError:
                        pass
            else:
                best_loss = ckpt.get("best_loss", float("inf"))

            del ckpt
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if start_step > t["steps"]:
                print(f"\n[done] Checkpoint is at step {saved_step}, target is {t['steps']}. "
                      f"Increase training.steps (e.g. --steps {saved_step + 5000}).")
                return
            print(f"[resume] Step {saved_step} | best train loss {best_loss:.4f} | "
                  f"best val loss {best_val_loss:.4f}\n")
        else:
            if fresh and latest_path.exists():
                print("\n[fresh] Ignoring existing checkpoint as requested.\n")
            else:
                print("\n[fresh] Starting from scratch.\n")

        start_time = time.time()
        start_tokens = tokens_seen
        last_log_time = start_time
        last_log_tokens = tokens_seen
        log_f = open(log_path, "a", buffering=1)

        def log_entry(entry):
            entry.setdefault("time", datetime.now(timezone.utc).isoformat())
            log_f.write(json.dumps(entry) + "\n")

        last_completed_step = start_step - 1
        best_improved_since_save = False

        def save_final_checkpoints():
            nonlocal best_val_loss, last_saved_checkpoint_step
            if last_completed_step < start_step:
                return
            if last_saved_checkpoint_step != last_completed_step:
                try:
                    trainer.save_checkpoint(latest_path, last_completed_step, best_loss, best_val_loss,
                                            data_signature, tokens_seen, model_config)
                    last_saved_checkpoint_step = last_completed_step
                except Exception as e:
                    print(f"[checkpoint] Could not save latest.pt during shutdown: {e}")

            if eval_interval == 0 and best_improved_since_save:
                try:
                    with trainer.ema_weights():
                        trainer.save_checkpoint(best_path, last_completed_step, best_loss, best_val_loss,
                                                data_signature, tokens_seen, model_config)
                except Exception as e:
                    print(f"[checkpoint] Could not save best.pt during shutdown: {e}")
            elif not best_path.exists():
                print("[checkpoint] No best checkpoint yet; running one evaluation to create one.")
                try:
                    with trainer.ema_weights():
                        v_loss = evaluate(model, val_tokens, cfg, device,
                                          trainer.blocks_fwd, trainer.heads_fwd)
                        best_val_loss = min(best_val_loss, v_loss)
                        trainer.save_checkpoint(best_path, last_completed_step, best_loss, best_val_loss,
                                                data_signature, tokens_seen, model_config)
                except Exception as e:
                    print(f"[checkpoint] Could not save best.pt during shutdown: {e}")

        try:
            for step in range(start_step, t["steps"] + 1):
                should_log = (step % t["log_interval"] == 0 or step == start_step or step == t["steps"])
                is_eval_step = ((eval_interval > 0 and step % eval_interval == 0)
                                or step == t["steps"])
                capture_loss = should_log or is_eval_step

                try:
                    if prefetcher is not None:
                        x, y = prefetcher.get()
                    else:
                        x, y = get_batch(train_tokens, t["batch_size"],
                                         cfg["model"]["max_len"], device)
                    final_loss, avg_loss, cur_lr = trainer.train_step(x, y, step,
                                                                      should_log=capture_loss)
                except RuntimeError as e:
                    if device.type == "cuda" and is_cuda_oom_error(e):
                        print(f"[oom] CUDA out of memory at step {step}; discarding this batch.")
                        trainer._zero_all_grads()
                        torch.cuda.empty_cache()
                        continue
                    raise

                tokens_seen += t["batch_size"] * cfg["model"]["max_len"]

                val_loss, val_ppl = None, None
                if is_eval_step:
                    with trainer.ema_weights():
                        val_loss = evaluate(model, val_tokens, cfg, device,
                                            trainer.blocks_fwd, trainer.heads_fwd)
                    val_ppl = perplexity(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        with trainer.ema_weights():
                            trainer.save_checkpoint(best_path, step, best_loss, best_val_loss,
                                                    data_signature, tokens_seen, model_config)

                if capture_loss and final_loss < best_loss:
                    best_loss = final_loss
                    best_improved_since_save = True

                if step % checkpoint_interval == 0 or step == t["steps"]:
                    trainer.save_checkpoint(latest_path, step, best_loss, best_val_loss,
                                            data_signature, tokens_seen, model_config)
                    last_saved_checkpoint_step = step
                    if eval_interval == 0 and best_improved_since_save:
                        with trainer.ema_weights():
                            trainer.save_checkpoint(best_path, step, best_loss, best_val_loss,
                                                    data_signature, tokens_seen, model_config)
                        best_improved_since_save = False

                if should_log:
                    now = time.time()
                    elapsed = now - start_time
                    cumulative_tokens = tokens_seen - start_tokens
                    cumulative_throughput = cumulative_tokens / max(elapsed, 1e-4)

                    window_elapsed = now - last_log_time
                    window_tokens = tokens_seen - last_log_tokens
                    instant_throughput = window_tokens / max(window_elapsed, 1e-4) if window_elapsed > 0 else cumulative_throughput

                    last_log_time = now
                    last_log_tokens = tokens_seen

                    vram = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
                    ppl_str = f" | Val PPL: {val_ppl:8.2f}" if val_ppl is not None else ""
                    if eval_interval > 0:
                        best_str = f"Best: {best_loss:.4f}/{best_val_loss:.4f}"
                    else:
                        best_str = f"Best: {best_loss:.4f}"

                    print(f"Step {step:05d}/{t['steps']} | Loss: {final_loss:.4f} (avg {avg_loss:.4f}) | "
                          f"{best_str}{ppl_str} | "
                          f"VRAM: {vram:.0f} MB | {instant_throughput:,.0f} tok/s (inst) | {cumulative_throughput:,.0f} tok/s (avg) | {elapsed:.0f}s")
                    log_entry({"step": step, "lr": cur_lr, "train_loss": final_loss,
                               "avg_stage_loss": avg_loss, "best_loss": best_loss,
                               "val_loss": val_loss, "val_ppl": val_ppl,
                               "best_val_loss": best_val_loss, "tokens_seen": tokens_seen,
                               "vram_mb": vram, "throughput_tok_s": instant_throughput})

                if step % sample_interval == 0 or step == t["steps"]:
                    with trainer.ema_weights():
                        for prompt in cfg["generation"]["prompts"]:
                            out = generate(model, tokenizer, prompt,
                                           cfg["generation"]["max_new_tokens"],
                                           cfg["generation"]["temperature"], cfg["generation"]["top_k"],
                                           cfg["generation"]["top_p"], cfg["generation"]["repetition_penalty"],
                                           device)
                            print(f"\n>>> [Step {step}] {prompt!r}\n    {out}\n")

                last_completed_step = step

        except KeyboardInterrupt:
            print("\n[interrupt] Caught KeyboardInterrupt; saving checkpoint before exiting...")
        finally:
            save_final_checkpoints()
            log_f.close()

        total_time = time.time() - start_time
        print("=" * 70)
        status = "complete" if last_completed_step == t["steps"] else "interrupted"
        print(f"Training {status} after {total_time:.1f}s | best train loss {best_loss:.4f} | "
              f"best val loss {best_val_loss:.4f} (ppl {perplexity(best_val_loss):.2f})")
        print(f"Latest: {latest_path} | Best: {best_path} | Log: {log_path}")
        print("=" * 70)
    finally:
        if prefetcher is not None:
            prefetcher.close()


if __name__ == "__main__":
    main()
