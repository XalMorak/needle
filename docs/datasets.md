## Combine free public datasets

The repository includes `scripts/combine_datasets.py` to prepare instruction
fine-tuning data without checking large datasets into Git. It uses the Hugging
Face `datasets` library, normalizes chat and instruction formats, removes exact
duplicates, shuffles deterministically, and writes JSONL compatible with local
fine-tuning workflows.

Install the optional dependency:

```bash
pip install -r requirements-local.txt
```

Prepare the default public datasets (OpenAssistant OASST1 and Databricks
Dolly 15k):

```bash
python scripts/combine_datasets.py \
  --output data/combined-instructions.jsonl \
  --max-per-dataset 10000 \
  --streaming
```

Use a different dataset or a named configuration by repeating `--dataset`:

```bash
python scripts/combine_datasets.py \
  --dataset OpenAssistant/oasst1 \
  --dataset databricks/databricks-dolly-15k \
  --dataset OWNER/REPO:CONFIG \
  --max-per-dataset 5000 \
  --output data/train.jsonl
```

The output contains `messages` and `source` fields. Review dataset licenses,
content, personally identifiable information, and usage terms before training
or redistributing a model. Dataset downloads and generated files are ignored
from source control; this integration does not bundle third-party data.
