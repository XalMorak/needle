# Needle Local

This repository includes the core Needle package and a small local-inference/data-prep workflow for running a public 13B GGUF model through `llama.cpp` and combining free instruction datasets into JSONL.

## Included utilities

- `scripts/run_13b.py`: runs a local 13B GGUF model through `llama.cpp`
- `scripts/combine_datasets.py`: merges public instruction datasets into a single JSONL file
- `requirements-local.txt`: install optional local tooling dependencies
- `docs/local-13b.md`: local runner setup notes

## Quick start

```bash
pip install -r requirements-local.txt
python scripts/combine_datasets.py --output data/combined-instructions.jsonl --streaming
python scripts/run_13b.py chat --prompt "Explain what this project does."
```
