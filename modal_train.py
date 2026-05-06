"""Modal wrapper to run train.py on a cloud GPU."""

import modal

app = modal.App("gemma-asr-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "curl")  # ffmpeg for torchcodec, git+curl for gguf install
    .pip_install("unsloth")
    .pip_install(
        "sentencepiece", "protobuf", "datasets>=4.3.0",
        "huggingface_hub>=0.34.0", "hf_transfer",
    )
    .pip_install("torchcodec==0.10.0")
    .pip_install("timm", extra_options="--no-deps --upgrade")
    .pip_install("jiwer")
    .pip_install("tensorboard>=2.18")
    .add_local_file("train.py", remote_path="/root/train.py")
)


@app.function(
    image=image,
    gpu="A100",
    timeout=10 * 3600,  # 10 hours (safety cap only; billing stops on completion)
    secrets=[modal.Secret.from_name("gemma-asr-secrets")],
)
def train():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "/root/train.py"],
        cwd="/root",
    )
    if result.returncode != 0:
        raise RuntimeError(f"train.py exited with code {result.returncode}")


@app.local_entrypoint()
def main():
    train.remote()
