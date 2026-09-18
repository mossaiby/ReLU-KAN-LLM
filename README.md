# Generalized Decoupled KAN-LLM

A high-throughput, memory-efficient implementation of an Autoregressive Causal Language Model featuring **Kolmogorov-Arnold Network (KAN) Feed-Forward Networks**, **Rotary Position Embeddings (RoPE)**, **Flash Attention (SDPA)**, and **Macro-Decoupled Blockwise Training**.

This project is explicitly engineered to train multi-million parameter language models locally on constrained hardware—such as **4 GB VRAM laptop GPUs (e.g., NVIDIA RTX 3050 Ti)**—while scaling seamlessly to workstation and datacenter GPUs.

---

## Key Architectural & Algorithmic Highlights

### 1. Macro-Decoupled Blockwise Training
Standard Transformer backpropagation requires caching all forward activations across all layers simultaneously until the loss is computed at the final layer. For a 4 GB GPU, this immediately causes Out-Of-Memory (OOM) errors.

This engine divides the model into sequential stages (e.g., 2 stages of 4 blocks each):
* **Stage 0:** Processes embeddings and initial blocks, projects through an intermediate local auxiliary head, computes cross-entropy loss, runs backpropagation, steps the Stage 0 optimizer, and **immediately purges gradients and backward graph buffers from VRAM**.
* **Stage 1:** Takes the detached hidden representations from Stage 0, processes the remaining blocks, projects through the final classification head, and updates Stage 1 independently.
* **Pruned Intermediate Heads:** Local heads are only instantiated at stage boundaries. Intermediate blocks use `nn.Identity()`, freeing >300 MB of VRAM and optimizer state on small GPUs.

### 2. Gated ReLUKAN Feed-Forward Networks
Instead of standard SwiGLU / MLP layers with static activations (GELU/SiLU), this model integrates **Kolmogorov-Arnold Network (KAN)** layers:
* Approximates continuous univariate functions using a learnable linear combination of non-linear basis splines:
  $$\text{basis}_k(x) = \text{ReLU}(\tanh(x) - \text{grid}_k)^2$$
* Combines a standard linear base branch with the non-linear KAN spline projection:
  $$y = W_{\text{base}} \cdot \text{SiLU}(x) + W_{\text{kan}} \cdot \text{basis}(x)$$
* Features a gated SwiGLU-style architecture (`dim` $\to$ `2 * dim` $\to$ `dim`) with layer normalization for training stability.

### 3. Rotary Position Embeddings (RoPE) & SDPA
* Employs PyTorch's native `F.scaled_dot_product_attention` (SDPA), triggering FlashAttention-2 or memory-efficient attention kernels automatically.
* RoPE tables are precomputed as persistent buffers and sliced via zero-copy views, avoiding Python dictionary overhead and eliminating `torch.compile` graph breaks.

### 4. Lightweight Compound EMA (Exponential Moving Average)
* Tracks active model weights in FP32 on a compound cadence (e.g., every 8 steps), reducing multi-tensor launch overhead by 87.5%.
* Features **in-VRAM memory swapping** during evaluation/sampling transitions, completing weight swaps in < 1 ms without round-trip PCIe transfers.
* Automatically saves smoothed EMA weights into `best.pt` when validation improves.

---

## Repository Structure

```
├── train_kan_llm.py        # Main training engine & data loading pipeline
├── train_advisor.py        # Live diagnostic & learning rate recommendation tool
├── config.example.json     # Hardware-tuned config for RTX 3050 Ti (4 GB VRAM)
├── requirements.txt        # CUDA 12.1 PyTorch dependencies
├── LICENSE                 # Apache 2.0 License
├── data/                   # Tokenizer vocabulary & token cache directory
└── checkpoints/            # Model checkpoints, logs, and eval records
```

---

## Installation & Setup

### 1. Prerequisites
* **OS:** Windows 10/11, Linux (Ubuntu 20.04+), or WSL2
* **Python:** 3.10 or 3.11
* **NVIDIA Driver:** Driver version $\ge 530$ supporting CUDA 12.1+

### 2. Create a Virtual Environment
```bash
# Linux / macOS / WSL2
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
Install dependencies using the configured PyTorch CUDA 12.1 wheel index:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify your GPU and CUDA availability:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('BF16 Supported:', torch.cuda.is_bf16_supported())"
```

---

## Quick Start

### Step 1: Initialize Configuration
Copy the hardware-tuned example config to `config.json`:
```bash
cp config.example.json config.json
```

### Step 2: Start Training
Launch the training pipeline:
```bash
python train_kan_llm.py
```

On its first run, the script will:
1. Stream and download the specified dataset split (e.g., FineWeb-Edu `sample-10BT`).
2. Train a Byte-Pair Encoding (BPE) tokenizer (vocab size 32,768) and cache it to `./data/`.
3. Pre-tokenize and cache training and validation token streams.
4. Pin token buffers in host RAM for zero-latency PCIe DMA transfers.
5. Initialize weights and begin macro-decoupled training.

### Step 3: Run Live Diagnostics
In a separate terminal, monitor learning curves, loss volatility, and receive dynamic learning rate recommendations:
```bash
python train_advisor.py
```

---

## Configuration Reference (`config.json`)

| Section | Parameter | Default / Recommended | Description |
| :--- | :--- | :--- | :--- |
| **`model`** | `dim` | `384` | Model hidden embedding dimension (must be divisible by 32). |
| | `num_layers` | `8` | Total number of fast decoupled transformer blocks. |
| | `max_len` | `256` | Maximum causal sequence length (context window). |
| | `k` | `6` | Number of spline knot basis intervals in ReLUKAN layers. |
| **`data`** | `type` | `"hf"` | Data source type: `"hf"` (HuggingFace) or `"text"` (local raw text file). |
| | `hf_name` | `"HuggingFaceFW/fineweb-edu"` | Dataset repository path. |
| | `hf_config`| `"sample-10BT"` | Specific dataset subset. **Use `sample-10BT` (28 GB), not `sample-100BT` (300 GB).** |
| | `max_articles` | `130000` | Limits processed articles (~115M tokens; matches a 25k-step run). |
| | `tokenizer_vocab_size` | `32768` | BPE vocabulary capacity. |
| | `val_fraction` | `0.01` | Fraction of tokens reserved for validation (1%). |
| **`training`**| `batch_size` | `16` | Micro-batch size per optimization step. |
| | `steps` | `25000` | Total training steps (~102.4M tokens at $B=16, L=256$). |
| | `stage_size` | `4` | Number of blocks grouped into each decoupled backward stage. |
| | `lr` | `0.00025` | Peak learning rate (reached at end of warmup). |
| | `min_lr` | `0.00002` | Minimum learning rate floor for cosine decay schedule. |
| | `warmup_steps` | `250` | Linear learning rate warmup steps (~1% of schedule). |
| | `weight_decay`| `0.0001` | AdamW weight decay. Keep $\le 0.001$ for flat param groups. |
| | `grad_clip` | `1.0` | Global gradient norm threshold (uses fused `foreach` kernels). |
| | `prefetch` | `true` | Runs background multi-threaded batch slicing and memory pinning. |
| | `ema_decay` | `0.9999` | Exponential moving average decay factor. |
| | `ema_device`| `"cuda"` | Device holding EMA shadow tensors (`"cuda"` or `"cpu"`). |
| | `checkpoint_interval`| `1000` | Frequency of saving atomic checkpoints to disk. |

### CLI Overrides
Any configuration option can be overridden at runtime without editing `config.json`:
```bash
# Resume with a custom learning rate and extended step count
python train_kan_llm.py --lr 1.8e-4 --steps 35000

# Train with torch.compile enabled (Linux / WSL2 recommended)
python train_kan_llm.py --compile --compile-mode default

# Train on a custom local text file
python train_kan_llm.py --data-type text --data-path ./my_corpus.txt --fresh
```

---

## Live Diagnostics with `train_advisor.py`

`train_advisor.py` inspects the JSONL logs generated by `train_kan_llm.py` and performs real-time linear regression, volatility analysis, and generalization checks.

### Example Terminal Output
```
========================================================================
               KAN-LLM TRAINING & HYPERPARAMETER REPORT                
========================================================================
Current Step:         14,000 / 25,000 (56.0%)
Active Learning Rate: 1.624e-04
Latest Train Loss:    3.2410 (ppl 25.56) | Best Train: 3.1904 (ppl 24.30)
Latest Val Loss:      3.2105 (ppl 24.79) | Best Val:   3.2105 (ppl 24.79)
Telemetry:            VRAM: 2,240 MB | Speed: 15,820 tok/s | Tokens Seen: 57.3M
ETA to Completion:    ~0:47:12 (11,000 steps remaining)
------------------------------------------------------------------------
[Stage Assessment] Stage 3 - Discourse Coherence & Broad Topic Semantics
  Description: Sentences are grammatically sound and form coherent paragraphs.
               Broad facts appear, though hallucinations are frequent.
------------------------------------------------------------------------
[Generalization Analysis]
  Generalization Gap (Val - Train): -0.0305
  HEALTHY: Validation loss is lower than training loss (EMA smoothing active).
------------------------------------------------------------------------
[Recent Trend] Based on last 14 training loss entries over 700 steps:
  Slope:          -1.42% change per 1,000 steps (-0.0461 loss/1k)
  Residual Noise: 1.28% (volatility around trendline)
------------------------------------------------------------------------
[Actionable Recommendations]
  • HEALTHY STEADY DESCENT: Steady optimization. Current hyperparameters are performing well.
------------------------------------------------------------------------
[Next Step CLI Command]
  To resume training with effective settings:
    python train_kan_llm.py --steps 25000
========================================================================
```

### Heuristic Stage Map
* **Stage 0 ($L \ge 10.4$):** Random initialization / uniform distribution over 32k vocabulary.
* **Stage 1 ($5.2 \le L < 10.4$):** Vocabulary, whitespace, and byte-level subword structure acquisition.
* **Stage 2 ($4.0 \le L < 5.2$):** Local syntax, parts of speech, and grammatical structure formation.
* **Stage 3 ($3.0 \le L < 4.0$):** Discourse coherence, punctuation structure, and broad topic semantics.
* **Stage 4 ($2.2 \le L < 3.0$):** Factual consolidation and domain terminology acquisition.
* **Stage 5 ($1.6 \le L < 2.2$):** Knowledge saturation; model extracts long-tail information.
* **Stage 6 ($L < 1.6$):** Compression floor / stylistic lock-in.

---

## Low-VRAM & Laptop GPU Tuning (RTX 3050 Ti Guide)

Running LLMs on mobile GPUs with 4 GB GDDR6 requires careful allocation:

1. **VRAM Budget Breakdown ($B=16, L=256, k=6$):**
   * Static Memory (Model + AdamW Stage States + EMA Shadow): **~977 MB**
   * Logits & Cross-Entropy Workspace ($16 \times 256 \times 32768$ in BF16): **~512 MB**
   * Forward Activations per 4-block stage: **~160 MB**
   * PyTorch CUDA Context & OS Display Window Manager: **~600–700 MB**
   * **Total Peak Allocation:** **~2.3–2.4 GB** (leaves ~1 GB safety margin to avoid OOM).

2. **Thermal & Power Management:**
   * Laptop GPUs thermal-throttle when reaching 75–80°C. Elevate the rear of the laptop by 1–2 inches to ensure adequate airflow.
   * On Windows, prevent automatic standby during training: *Settings $\to$ System $\to$ Power & Battery $\to$ Sleep: Never*.
   * Disable hardware acceleration in web browsers (Chrome/Edge/Discord can consume 300–500 MB of VRAM).

3. **FineWeb-Edu Dataset Warning:**
   * Always use subset `"sample-10BT"` (28.5 GB). Never use `"sample-100BT"`, which attempts to download over 300 GB of Parquet shards.

---

## Checkpoint Architecture

All checkpoints are saved atomically using a temporary file exchange (`.pt.tmp` $\to$ `.pt` via `os.replace` and `fsync`), preventing checkpoint corruption if interrupted mid-save:

* **`kan_model_latest.pt`:** Contains raw training weights, individual optimizer states for all stages, random number generator states (CPU, CUDA, Python, prefetcher), and AMP scaler states. Used to resume training seamlessly.
* **`kan_model_best.pt`:** Contains the model weights reflecting the lowest validation loss (saved with smoothed **EMA weights** in `model_state`). Ideal for inference and downstream evaluation.
* **`kan_model_log.jsonl`:** Append-only structured JSON log recording step-by-step metrics, VRAM consumption, loss values, and throughput telemetry.

---

## License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.
