## Local 13B model

Needle itself is an 8–29 MB tiny-device model. It does not contain a 13B
parameter architecture or weights. For users who need a full local language
model, this repository now includes an optional `llama.cpp` runner. It uses
GGUF quantization, so the default Q4_K_M file is approximately 8 GB rather
than 13B × 4 bytes of RAM.

### Install

Build `llama.cpp` from its official repository and make `llama-cli` and
`llama-server` available on `PATH`:

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_NATIVE=ON
cmake --build llama.cpp/build --config Release -j 4
export PATH="$PWD/llama.cpp/build/bin:$PATH"

pip install -r requirements-local.txt
```

The runner downloads the default quantized 13B model on first use. Model
weights are cached outside the repository:

```bash
python scripts/run_13b.py chat --prompt "Explain what this project does."
```

For a CPU-only run, keep `--gpu-layers 0`. With a CUDA/Metal/Vulkan build,
move layers to the GPU, for example:

```bash
python scripts/run_13b.py --gpu-layers 40 chat \
  --prompt "Write a short Python example."
```

Start an OpenAI-compatible local HTTP server:

```bash
python scripts/run_13b.py server --port 8080
```

Then call `http://127.0.0.1:8080/v1/chat/completions`. To use another
13B GGUF model, provide its Hugging Face repository and exact file name:

```bash
python scripts/run_13b.py \
  --model-repo OWNER/REPO \
  --model-file MODEL.Q4_K_M.gguf \
  chat --prompt "Hello"
```

This is an inference runner, not a pre-trained model checkpoint. Training a
new 13B model from scratch requires a large curated dataset and multi-GPU
compute; fine-tuning an existing open model is the practical next step.
