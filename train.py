# -*- coding: utf-8 -*-
"""Gemma 4 ASR fine-tuning with Unsloth + LoRA. Edit the config block below to use your language and datasets."""

import os
from datetime import datetime

HF_TOKEN_DATASETS = os.environ["HF_TOKEN_DATASETS"]
HF_TOKEN_MODELS = os.environ["HF_TOKEN_MODELS"]

# All audio samples are ≤30s. At 16kHz, 30s = 480K samples → ~1500 audio tokens.
# With prompt + text overhead (~200 tokens), max total ≈ 1700. 4096 ensures zero truncation.
MAX_SEQ_LENGTH = 4096

if not HF_TOKEN_DATASETS:
    raise RuntimeError("HF_TOKEN_DATASETS environment variable is not set")
if not HF_TOKEN_MODELS:
    raise RuntimeError("HF_TOKEN_MODELS environment variable is not set")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Config ───────────────────────────────────────────────────────────────────
LANGUAGE = "Punjabi"        # Human-readable language name used in prompts, e.g. "Hindi", "Tamil"

SMOKE_TEST_SAMPLES = 0      # 0 = full dataset; set e.g. 50 for a quick smoke test
NUM_TRAIN_EPOCHS = 3
MODEL_NAME = "unsloth/gemma-4-E2B-it"  # or gemma-4-E4B-it / gemma-4-26B-A4B-it
LORA_RANK = 8               # 8, 16, 32, 64

# HF dataset repo names (just the name — username is resolved from HF_TOKEN_DATASETS).
# Dataset must have an `audio` column (16kHz) and a transcript column (set TRANSCRIPT_COLUMN below).
TRAIN_DATASET_NAMES = ["your_train_dataset_1", "your_train_dataset_2"]
TEST_DATASET_NAMES  = ["your_test_dataset_1", "your_test_dataset_2"]
TRANSCRIPT_COLUMN   = "sentence"  # column name for ground truth text — "sentence", "text", "transcription", etc.

ASR_SYSTEM_PROMPT = f"You are an assistant that transcribes {LANGUAGE} speech accurately."
ASR_USER_PROMPT = f"Please transcribe this {LANGUAGE} audio."

# ── Resolve HF usernames from tokens ─────────────────────────────────────────
from huggingface_hub import HfApi
datasets_hf_user = HfApi(token=HF_TOKEN_DATASETS).whoami()["name"]
models_hf_user = HfApi(token=HF_TOKEN_MODELS).whoami()["name"]
print(f"Datasets token user (HF_TOKEN_DATASETS): {datasets_hf_user}")
print(f"Models token user (HF_TOKEN_MODELS): {models_hf_user}")

TRAIN_DATASETS = [f"{datasets_hf_user}/{name}" for name in TRAIN_DATASET_NAMES]
TEST_DATASETS  = [f"{datasets_hf_user}/{name}" for name in TEST_DATASET_NAMES]
print(f"Train datasets: {TRAIN_DATASETS}")
print(f"Test datasets:  {TEST_DATASETS}")

# ── Imports ──────────────────────────────────────────────────────────────────
import torch
torch._dynamo.config.recompile_limit = 64

from unsloth import FastModel
from transformers import TextStreamer
from datasets import load_dataset, concatenate_datasets, Audio
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import jiwer
_model_size_tag = MODEL_NAME.split("/")[-1].replace("gemma-4-", "").replace("-it", "").lower()

# ── Text Preprocessing ───────────────────────────────────────────────────────
# Replace this with any domain-specific normalisation your dataset needs.
def normalize_transcript(text):
    return text.strip()

# ── Metric Helpers ────────────────────────────────────────────────────────────
def compute_all_metrics(references, predictions):
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    ccer_transform = jiwer.Compose([jiwer.RemoveWhiteSpace()])
    ccer = jiwer.cer(ccer_transform(references), ccer_transform(predictions))
    return {"wer": wer, "cer": cer, "ccer": ccer}

# ── Load Model ───────────────────────────────────────────────────────────────
# Unsloth ships transformers 5.5.0 which calls normal_() on ALL missing keys
# including quantized uint8 params → crash. Patch the transformers init wrapper
# to skip uint8, so float missing keys (k_norm, lm_head) still init properly.
import transformers.initialization as _tf_init
_orig_tf_normal = _tf_init.TORCH_INIT_FUNCTIONS["normal_"]
def _safe_tf_normal(tensor, mean=0., std=1., generator=None):
    if tensor.dtype == torch.uint8:
        return tensor
    return _orig_tf_normal(tensor, mean=mean, std=std, generator=generator)
_tf_init.TORCH_INIT_FUNCTIONS["normal_"] = _safe_tf_normal

model, processor = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    dtype=torch.bfloat16,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    full_finetuning=False,
    token=HF_TOKEN_DATASETS,
)
_tf_init.TORCH_INIT_FUNCTIONS["normal_"] = _orig_tf_normal  # restore after load

# ── Helpers ──────────────────────────────────────────────────────────────────
import numpy as np

def pad_audio(audio_array):
    """Pad audio to a multiple of 128 samples so that the processor's token
    count matches the feature extractor's output (which uses pad_to_multiple_of=128)."""
    n = len(audio_array)
    remainder = n % 128
    if remainder != 0:
        audio_array = np.pad(audio_array, (0, 128 - remainder))
    return audio_array

def do_inference(messages, max_new_tokens=256):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids = output[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated_ids, skip_special_tokens=True).replace("<turn|>", "").strip()
    print(text)
    del inputs, output, generated_ids
    torch.cuda.empty_cache()

def get_transcription(messages, max_new_tokens=256):
    """Run inference and return the generated text (no streaming)."""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Decode only the newly generated tokens
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated_ids, skip_special_tokens=True).replace("<turn|>", "").strip()
    # Free GPU tensors immediately
    del inputs, output_ids, generated_ids
    torch.cuda.empty_cache()
    return text

# ── Load Datasets ────────────────────────────────────────────────────────────
print(f"\n=== Loading {len(TRAIN_DATASETS)} training dataset(s) ===")
full_dataset = concatenate_datasets([
    load_dataset(td, split="train", token=HF_TOKEN_DATASETS) for td in TRAIN_DATASETS
])
total = len(full_dataset)
full_dataset = full_dataset.cast_column("audio", Audio(sampling_rate=16000))

full_dataset = full_dataset.shuffle(seed=42)
if SMOKE_TEST_SAMPLES:
    dataset = full_dataset.select(range(min(SMOKE_TEST_SAMPLES, total)))
    print(f"Full dataset: {total} samples | Smoke test: {len(dataset)} samples")
else:
    print(f"Full dataset: {total} samples (using all)")
    dataset = full_dataset

# Preprocess text
def preprocess_text(batch):
    batch[TRANSCRIPT_COLUMN] = normalize_transcript(batch[TRANSCRIPT_COLUMN])
    return batch

print("Preprocessing training text...")
dataset = dataset.map(preprocess_text)

# Load test datasets
print(f"\n=== Loading {len(TEST_DATASETS)} test datasets ===")
test_dataset = concatenate_datasets([
    load_dataset(td, split="train+validation+test", token=HF_TOKEN_DATASETS) for td in TEST_DATASETS
])
print(f"Test dataset: {len(test_dataset)} samples")
test_dataset = test_dataset.map(preprocess_text)
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

if SMOKE_TEST_SAMPLES:
    test_dataset = test_dataset.select(range(min(SMOKE_TEST_SAMPLES, len(test_dataset))))
    print(f"Test subset: {len(test_dataset)} samples")

# Reserve a few samples for inference demo
test_samples = [test_dataset[i] for i in range(min(3, len(test_dataset)))]

for i, ts in enumerate(test_samples):
    sr = ts["audio"]["sampling_rate"]
    dur = len(ts["audio"]["array"]) / sr
    print(f"Test sample {i}: sr={sr}, duration={dur:.1f}s, ground truth: {ts[TRANSCRIPT_COLUMN]}")

# ── Baseline Inference ───────────────────────────────────────────────────────
print("\n=== Baseline (before fine-tuning) ===")
for i, ts in enumerate(test_samples):
    print(f"\n--- Test sample {i} ---")
    print(f"Ground truth: {ts['sentence']}")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": ASR_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": ts["audio"]["array"]},
                {"type": "text", "text": ASR_USER_PROMPT},
            ],
        },
    ]
    try:
        do_inference(messages)
    except Exception as e:
        print(f"Inference failed: {e}")

# ── Apply LoRA ───────────────────────────────────────────────────────────────
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_RANK,
    lora_alpha=LORA_RANK * 2,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    target_modules=[
        # Text decoder - attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Text decoder - MLP
        "gate_proj", "up_proj", "down_proj",
        # Text decoder - PLE (Per-Layer Embedding)
        "per_layer_input_gate", "per_layer_projection",
        # Audio encoder - output projection
        "output_proj",
        # Audio-to-text embedding projection
        "embedding_projection",
        # Audio encoder - conformer attention
        "relative_k_proj", "input_proj_linear",
    ],
)


# ── Data Prep ────────────────────────────────────────────────────────────────
def format_asr_data(samples: dict) -> dict[str, list]:
    formatted = {"messages": []}
    for idx in range(len(samples["audio"])):
        audio = samples["audio"][idx]["array"]
        label = str(samples[TRANSCRIPT_COLUMN][idx])
        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": ASR_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": ASR_USER_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]
        formatted["messages"].append(message)
    return formatted

dataset = dataset.map(format_asr_data, batched=True, batch_size=4, num_proc=4)

# Store eval results for model card
epoch_eval_results = []

# ── Training ─────────────────────────────────────────────────────────────────
_batch_size = 16
_steps_per_epoch = max(1, len(dataset) // _batch_size)
_log_save_steps = max(1, _steps_per_epoch // 5)
print(f"\n=== Training: {NUM_TRAIN_EPOCHS} epochs, {len(dataset)} train samples, {len(test_dataset)} test samples, logging/saving every {_log_save_steps} steps ===")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=SFTConfig(
        per_device_train_batch_size=_batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=5e-5,
        bf16=True,
        warmup_ratio=0.05,
        logging_strategy="steps",
        logging_steps=_log_save_steps,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to=["tensorboard"],
        logging_dir="outputs/runs",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_SEQ_LENGTH,
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
print(f"\n{trainer_stats.metrics['train_runtime']:.1f} seconds used for training.")
print(f"{trainer_stats.metrics['train_runtime']/60:.2f} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")

# ── Post-Training Evaluation ──────────────────────────────────────────────────
def build_asr_messages(audio_array):
    audio_array = pad_audio(audio_array)
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": ASR_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": ASR_USER_PROMPT},
            ],
        },
    ]

# ── Post-Training: use last epoch eval results ───────────────────────────────
# The callback already ran eval at each epoch. Use the last epoch's metrics.
if epoch_eval_results:
    metrics = epoch_eval_results[-1]
    print(f"\n=== Final Eval Metrics (epoch {metrics['epoch']}) ===")
    for name in ["wer", "cer", "ccer"]:
        print(f"  {name.upper()}: {metrics[name]:.4f}")
else:
    # Fallback: run eval now if callback didn't fire
    print(f"\n=== Evaluating on {len(test_dataset)} test samples ===")
    all_references = []
    all_predictions = []
    for i in range(len(test_dataset)):
        ts = test_dataset[i]
        ref = ts[TRANSCRIPT_COLUMN]
        messages = build_asr_messages(ts["audio"]["array"])
        try:
            pred = get_transcription(messages)
        except Exception as e:
            print(f"  Sample {i} inference failed: {e}")
            pred = ""
        all_references.append(ref)
        all_predictions.append(pred)
    metrics = compute_all_metrics(all_references, all_predictions)
    metrics["epoch"] = NUM_TRAIN_EPOCHS
    metrics["step"] = trainer.state.global_step
    epoch_eval_results.append(metrics)
    print(f"\n=== Eval Metrics ===")
    for name in ["wer", "cer", "ccer"]:
        print(f"  {name.upper()}: {metrics[name]:.4f}")

# Consolidate TensorBoard logs: copy trainer's standard logs + add eval metrics
try:
    from torch.utils.tensorboard import SummaryWriter
    import glob, shutil

    tb_consolidated_dir = "outputs/tb_logs"
    os.makedirs(tb_consolidated_dir, exist_ok=True)

    # Copy trainer's standard TB event files (loss, lr, etc.) into consolidated dir
    trainer_tb_dir = "outputs/runs"
    if os.path.isdir(trainer_tb_dir):
        for f in glob.glob(os.path.join(trainer_tb_dir, "**", "events.out.tfevents.*"), recursive=True):
            shutil.copy2(f, tb_consolidated_dir)
        print(f"Copied trainer TB logs from {trainer_tb_dir} to {tb_consolidated_dir}")

    # Add eval metrics to the same consolidated dir
    tb_writer = SummaryWriter(log_dir=tb_consolidated_dir)
    for result in epoch_eval_results:
        for name in ["wer", "cer", "ccer"]:
            tb_writer.add_scalar(f"eval/{name}", result[name], global_step=result["step"])
    tb_writer.close()
    print("All metrics (standard + eval) written to TensorBoard.")
except ImportError:
    print("TensorBoard not available, skipping TB log consolidation.")

# ── Save & Upload to Hugging Face ────────────────────────────────────────────
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
train_ds_name = "+".join(TRAIN_DATASET_NAMES)
REPO_NAME = f"gemma-4-{_model_size_tag}-r{LORA_RANK}-{train_ds_name}-epochs-{NUM_TRAIN_EPOCHS}-test-datasets-{len(TEST_DATASETS)}-{timestamp}"
if SMOKE_TEST_SAMPLES:
    REPO_NAME += f"-smoke-{SMOKE_TEST_SAMPLES}"
LOCAL_DIR = REPO_NAME
HF_REPO = f"{models_hf_user}/{REPO_NAME}"

print(f"\n=== Saving LoRA adapters to {LOCAL_DIR} ===")
model.save_pretrained(LOCAL_DIR)
processor.save_pretrained(LOCAL_DIR)

# ── Generate Model Card ──────────────────────────────────────────────────────
# Build combined training results table (like fb_mms_1b's auto-generated card)
# Merge training loss entries with eval metrics by epoch
loss_by_epoch = {}
for entry in trainer.state.log_history:
    if "loss" in entry:
        epoch_key = int(entry.get("epoch", 0))
        loss_by_epoch[epoch_key] = {
            "loss": entry.get("loss", ""),
            "step": entry.get("step", ""),
            "learning_rate": entry.get("learning_rate", ""),
        }

eval_by_epoch = {int(r["epoch"]): r for r in epoch_eval_results}
all_epochs = sorted(set(list(loss_by_epoch.keys()) + list(eval_by_epoch.keys())))

combined_table_rows = ""
for ep in all_epochs:
    loss_info = loss_by_epoch.get(ep, {})
    eval_info = eval_by_epoch.get(ep, {})
    row_loss = loss_info.get("loss", "")
    row_step = loss_info.get("step", eval_info.get("step", ""))
    row_epoch = float(ep)
    row_wer = f"{eval_info['wer']:.4f}" if eval_info else ""
    row_cer = f"{eval_info['cer']:.4f}" if eval_info else ""
    row_ccer = f"{eval_info['ccer']:.4f}" if eval_info else ""
    combined_table_rows += f"| {row_epoch} | {row_loss} | {row_ccer} | {row_cer} | {row_wer} |\n"

train_loss = trainer_stats.metrics.get("train_loss", "N/A")
train_runtime = trainer_stats.metrics.get("train_runtime", 0)
train_samples = len(dataset)
test_samples_count = len(test_dataset)
subset_note = f" (smoke test: {SMOKE_TEST_SAMPLES} samples)" if SMOKE_TEST_SAMPLES else ""

# Get framework versions
import transformers as _transformers
import datasets as _datasets
import tokenizers as _tokenizers
fw_transformers = _transformers.__version__
fw_torch = torch.__version__
fw_datasets = _datasets.__version__
fw_tokenizers = _tokenizers.__version__

_language_tag = LANGUAGE.lower().replace(" ", "-")
model_card = f"""---
library_name: peft
base_model: {MODEL_NAME}
tags:
  - {_language_tag}
  - asr
  - lora
  - gemma4
  - unsloth
  - speech-recognition
pipeline_tag: automatic-speech-recognition
---

# Gemma 4 {_model_size_tag.upper()} - {LANGUAGE} ASR (LoRA)

Fine-tuned [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) on {LANGUAGE} speech data using LoRA + Unsloth.

## Training Details

| Parameter | Value |
|---|---|
| Base model | {MODEL_NAME} |
| Train datasets | {", ".join(TRAIN_DATASETS)}{subset_note} |
| Train samples | {train_samples} |
| Test samples | {test_samples_count} |
| Epochs | {NUM_TRAIN_EPOCHS} |
| Batch size | {trainer.args.per_device_train_batch_size} x {trainer.args.gradient_accumulation_steps} grad accum = {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps} effective |
| Learning rate | 5e-5 (cosine) |
| LoRA r/alpha | {LORA_RANK}/{LORA_RANK * 2} |
| Trainable params | 23.9M (0.30%) |
| Precision | FP16 + 4-bit quantization |
| Avg train loss | {train_loss} |
| Training time | {train_runtime:.1f}s |

## Training results

Evaluated on {test_samples_count} test samples from {len(TEST_DATASETS)} test dataset(s).

| Epoch | Training Loss | ccer | cer | wer |
|---|---|---|---|---|
{combined_table_rows}
### Final Metrics (Epoch {metrics['epoch']})

| Metric | Value |
|---|---|
| **WER** (Word Error Rate) | {metrics['wer']:.4f} |
| **CER** (Character Error Rate) | {metrics['cer']:.4f} |
| **CCER** (CER without spaces) | {metrics['ccer']:.4f} |

## Test Datasets
{chr(10).join(f'- {td}' for td in TEST_DATASETS)}

## Framework versions
- Transformers {fw_transformers}
- Pytorch {fw_torch}
- Datasets {fw_datasets}
- Tokenizers {fw_tokenizers}
"""

readme_path = os.path.join(LOCAL_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(model_card)
print(f"Model card written to {readme_path}")

print(f"\n=== Uploading to Hugging Face: {HF_REPO} ===")
model.push_to_hub(HF_REPO, token=HF_TOKEN_MODELS, private=True)
processor.push_to_hub(HF_REPO, token=HF_TOKEN_MODELS, private=True)

# Push model card + TensorBoard logs
from huggingface_hub import HfApi
hf_api = HfApi(token=HF_TOKEN_MODELS)

hf_api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=HF_REPO,
    repo_type="model",
)
print("Model card uploaded to HF!")

tb_logs_dir = "outputs/tb_logs"
if os.path.isdir(tb_logs_dir):
    hf_api.upload_folder(
        folder_path=tb_logs_dir,
        path_in_repo="logs",
        repo_id=HF_REPO,
        repo_type="model",
    )
    print("TensorBoard logs uploaded to HF!")
print("Upload complete!")

# ── Convert LoRA adapter to GGUF ─────────────────────────────────────────────
print("\n=== Converting LoRA adapter to GGUF ===")

GGUF_REPO = f"{models_hf_user}/{REPO_NAME}-gguf"
GGUF_FILE = os.path.join(LOCAL_DIR, "lora-adapter.gguf")

try:
    import subprocess as _sp
    import sys as _sys

    # Save base model config.json (needed by LoRA-to-GGUF converter)
    import json as _json
    _config_dict = _json.loads(model.config.to_json_string())
    _config_dict.pop("quantization_config", None)
    with open(os.path.join(LOCAL_DIR, "config.json"), "w") as _f:
        _json.dump(_config_dict, _f, indent=2)

    # Download convert_lora_to_gguf.py from llama.cpp
    _scripts_dir = "/tmp/llama_cpp_scripts"
    os.makedirs(_scripts_dir, exist_ok=True)
    _sp.run([_sys.executable, "-m", "pip", "install", "-q",
             "gguf @ git+https://github.com/ggml-org/llama.cpp#subdirectory=gguf-py"], check=True)
    for _script in ["convert_lora_to_gguf.py", "convert_hf_to_gguf.py"]:
        _url = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/master/{_script}"
        _sp.run(["curl", "-sL", _url, "-o", os.path.join(_scripts_dir, _script)], check=True)

    _result = _sp.run(
        [_sys.executable, os.path.join(_scripts_dir, "convert_lora_to_gguf.py"),
         "--base", LOCAL_DIR, LOCAL_DIR, "--outfile", GGUF_FILE],
        capture_output=True, text=True,
    )
    print(_result.stdout[-2000:] if _result.stdout else "")
    if _result.returncode != 0:
        print(f"LoRA GGUF conversion failed: {_result.stderr[-2000:]}")
    else:
        print(f"\nLoRA GGUF exported to {GGUF_FILE}")

        print(f"\n=== Uploading LoRA GGUF to {GGUF_REPO} ===")
        hf_api.create_repo(GGUF_REPO, repo_type="model", private=True, exist_ok=True)
        hf_api.upload_file(
            path_or_fileobj=GGUF_FILE,
            path_in_repo="lora-adapter.gguf",
            repo_id=GGUF_REPO,
            repo_type="model",
        )
        print(f"LoRA GGUF uploaded to {GGUF_REPO}!")
except Exception as e:
    print(f"LoRA GGUF conversion failed (non-fatal): {e}")
