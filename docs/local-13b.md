## Local 13B model + public data

This repository includes an optional local workflow for public 13B GGUF inference and common public instruction-data preparation.

### Install

```bash
pip install -r requirements-local.txt
```

### Build `llama.cpp`

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_NATIVE=ON
cmake --build llama.cpp/build --config Release -j 4
export PATH="$PWD/llama.cpp/build/bin:$PATH"
```

### Run a local 13B GGUF model

```bash
python scripts/run_13b.py chat --prompt "Explain what this project does."
```

### Combine public instruction datasets

```bash
python scripts/combine_datasets.py --output data/combined-instructions.jsonl --streaming
```

### Custom dataset list

```bash
python scripts/combine_datasets.py \
  --dataset OpenAssistant/oasst1 \
  --dataset databricks/databricks-dolly-15k \
  --dataset OWNER/REPO:CONFIG \
  --output data/train.jsonl
```
