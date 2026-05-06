# gemma-asr-finetune

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co)
[![Unsloth](https://img.shields.io/badge/⚡-Unsloth-purple)](https://github.com/unslothai/unsloth)
[![Modal](https://img.shields.io/badge/☁️-Modal-blue)](https://modal.com)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)

Fine-tune [Gemma 4](https://huggingface.co/unsloth/gemma-4-E2B-it) for speech recognition (ASR) on any language, using LoRA + [Unsloth](https://github.com/unslothai/unsloth), with one-command cloud training on [Modal](https://modal.com).

## The problem

Most production ASR systems ([Whisper](https://github.com/openai/whisper), [MMS](https://huggingface.co/facebook/mms-1b-all)) are trained on majority languages. For low-resource languages — regional dialects, liturgical speech, minority languages — they produce poor results or fail entirely. The gap isn't the model architecture; it's the absence of fine-tuning on domain-specific audio.

[Gemma 4](https://huggingface.co/collections/google/gemma-4-release-680eb1d6e63f14172b52c0a6)'s multimodal architecture accepts raw audio directly, making it a strong candidate for ASR fine-tuning without a separate acoustic encoder. But the tooling to actually do this — combine [Unsloth](https://github.com/unslothai/unsloth)'s efficient LoRA, a cloud GPU, evaluation, and model export — is scattered across docs and forum posts.

This repo gives you a single, working script.

## What you get

- Fine-tune [Gemma 4 E2B or E4B](https://huggingface.co/unsloth/gemma-4-E2B-it) on your audio dataset with 4-bit QLoRA (fits on an A10G or A100)
- Automatic evaluation after each epoch: WER, CER, CCER
- LoRA adapter + [GGUF](https://huggingface.co/docs/hub/en/gguf) export uploaded to [HuggingFace](https://huggingface.co) with a generated model card
- TensorBoard logging
- One-command cloud training via [Modal](https://modal.com) — no local GPU needed
- Works with any [HuggingFace audio dataset](https://huggingface.co/docs/datasets/audio_load)

## Requirements

- A [Modal](https://modal.com) account (free tier covers short runs)
- A [HuggingFace](https://huggingface.co) account with your audio dataset uploaded
## Dataset format

Your [HuggingFace dataset](https://huggingface.co/docs/datasets/audio_load) needs two columns:

| Column | Type | Description |
|--------|------|-------------|
| `audio` | [`Audio`](https://huggingface.co/docs/datasets/v2.14.0/en/package_reference/main_classes#datasets.Audio) (16kHz) | Audio samples |
| `sentence` (or any name) | `string` | Ground truth transcript |

The `audio` column name is the HuggingFace standard. The transcript column name is configurable — set `TRANSCRIPT_COLUMN` in the config block.

If your dataset uses a different audio sampling rate, the script resamples it to 16kHz automatically.

## Installation

Training runs entirely on [Modal](https://modal.com) — no local GPU or CUDA setup needed. You only need Modal installed locally.

```bash
pip install modal
modal setup   # authenticates your Modal account
```

[Unsloth](https://github.com/unslothai/unsloth), PyTorch, and all other training dependencies are installed automatically inside the Modal container when you run the job.

## Setup

### 1. Create a Modal secret

In the [Modal dashboard](https://modal.com/secrets), create a secret named `gemma-asr-secrets` with:

| Key | Value |
|-----|-------|
| `HF_TOKEN_DATASETS` | [HuggingFace token](https://huggingface.co/settings/tokens) with **read** access to your datasets |
| `HF_TOKEN_MODELS` | [HuggingFace token](https://huggingface.co/settings/tokens) with **write** access (to push trained models) |

### 2. Edit the config block in `train.py`

```python
# ── Config ───────────────────────────────────────────────────────────────────
LANGUAGE          = "Punjabi"       # used in prompts — set to your language
TRANSCRIPT_COLUMN = "sentence"    # transcript column name in your dataset

SMOKE_TEST_SAMPLES   = 0          # 0 = full dataset; set e.g. 50 for a quick smoke test
NUM_TRAIN_EPOCHS     = 3
MODEL_NAME           = "unsloth/gemma-4-E2B-it"  # or gemma-4-E4B-it
LORA_RANK            = 8          # 8, 16, 32, 64

TRAIN_DATASET_NAMES  = ["your_train_dataset_1", "your_train_dataset_2"]
TEST_DATASET_NAMES   = ["your_test_dataset_1", "your_test_dataset_2"]
```

Dataset names are the repo name only — the username is resolved automatically from `HF_TOKEN_DATASETS`.

### 3. Run

```bash
modal run modal_train.py
```

Training runs on a cloud A100. When it finishes, the LoRA adapter is uploaded to your [HuggingFace](https://huggingface.co) account as a private repo, along with a model card and TensorBoard logs.

## GPU targets

| GPU | VRAM | Works with |
|-----|------|-----------|
| A100 (default) | 40GB | E2B, E4B |
| A10G | 24GB | E2B with `LORA_RANK=8` |

Change `gpu="A100"` in `modal_train.py` to target a different GPU.

## Text preprocessing

`train.py` includes a `normalize_transcript()` stub — replace it with any domain-specific normalisation your dataset needs (strip separators, expand abbreviations, standardise number words, etc.). For clean datasets the default `text.strip()` is sufficient.

## Metrics

| Metric | Description |
|--------|-------------|
| WER | Word error rate |
| CER | Character error rate |
| CCER | CER after removing whitespace |

## Output

After training you'll find in your [HuggingFace](https://huggingface.co) account:

- `{your-username}/gemma-4-e2b-r8-{dataset}-epochs-{n}-{timestamp}` — LoRA adapter (safetensors)
- `{your-username}/gemma-4-e2b-r8-{dataset}-epochs-{n}-{timestamp}-gguf` — GGUF adapter for use with [llama.cpp](https://github.com/ggml-org/llama.cpp)

## Using your trained model

### GGUF with llama.cpp (recommended for local inference)

**1. Install llama.cpp**

```bash
# macOS
brew install llama.cpp

# or build from source
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -t llama-mtmd-cli
```

**2. Download the base model and mmproj**

```bash
huggingface-cli download unsloth/gemma-4-E2B-it-GGUF \
  --include "gemma-4-E2B-it-Q8_0.gguf" "mmproj-gemma-4-E2B-it-Q8_0.gguf" \
  --local-dir ./models
```

**3. Download your LoRA adapter**

```bash
huggingface-cli download your-username/your-lora-gguf-repo \
  --include "lora-adapter.gguf" \
  --local-dir ./models
```

**4. Run inference**

```bash
llama-mtmd-cli \
  -m models/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj models/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  --lora models/lora-adapter.gguf \
  --audio your_audio.wav \
  -p "Please transcribe this audio." \
  --temp 0 -n 256 --jinja
```

### LM Studio

[LM Studio](https://lmstudio.ai) supports GGUF models with LoRA adapters. Load the base Gemma 4 model and attach your LoRA adapter from the model settings panel.

### Python (HuggingFace + PEFT)

```python
from transformers import AutoProcessor
from peft import PeftModel
from unsloth import FastModel

model, processor = FastModel.from_pretrained(
    "unsloth/gemma-4-E2B-it",
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "your-username/your-lora-repo")
model.eval()
```
