"""Download and combine public Hugging Face instruction datasets into JSONL.

No dataset files are committed to the repository.  This script downloads only
metadata/examples requested by the user and writes a local training file.
Always review each dataset's license and terms before commercial use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DEFAULT_DATASETS = (
    "OpenAssistant/oasst1",
    "databricks/databricks-dolly-15k",
)


def _messages(row: dict) -> list[dict] | None:
    raw = row.get("messages") or row.get("conversation")
    if isinstance(raw, list):
        result = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or item.get("from")
            content = item.get("content") or item.get("value") or item.get("text")
            role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
            if role in {"system", "user", "assistant"} and isinstance(content, str) and content.strip():
                result.append({"role": role, "content": content.strip()})
        return result if any(m["role"] == "user" for m in result) else None

    instruction = row.get("instruction") or row.get("prompt") or row.get("question")
    answer = row.get("response") or row.get("output") or row.get("answer")
    if isinstance(instruction, str) and instruction.strip() and isinstance(answer, str) and answer.strip():
        user = instruction.strip()
        if isinstance(row.get("input"), str) and row["input"].strip():
            user += "\n\n" + row["input"].strip()
        return [{"role": "user", "content": user}, {"role": "assistant", "content": answer.strip()}]
    return None


def _fingerprint(messages: list[dict]) -> str:
    text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_dataset(spec: str) -> tuple[str, str | None]:
    repo, separator, config = spec.partition(":")
    return repo, config or None


def combine(datasets: list[str], output: Path, max_per_dataset: int, seed: int, streaming: bool) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install local training dependencies: pip install -r requirements-local.txt") from exc

    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()
    for spec in datasets:
        repo, config = _parse_dataset(spec)
        kwargs = {"path": repo, "split": "train", "streaming": streaming}
        if config:
            kwargs["name"] = config
        source = load_dataset(**kwargs)
        selected = []
        for row in source:
            messages = _messages(dict(row))
            if not messages:
                continue
            key = _fingerprint(messages)
            if key in seen:
                continue
            seen.add(key)
            selected.append({"messages": messages, "source": repo})
            if len(selected) >= max_per_dataset:
                break
        rng.shuffle(selected)
        rows.extend(selected)

    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Combine free public HF instruction datasets")
    p.add_argument("--dataset", action="append", dest="datasets", help="REPO[:CONFIG]; repeatable")
    p.add_argument("--output", default="data/combined-instructions.jsonl")
    p.add_argument("--max-per-dataset", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--streaming", action="store_true", help="Avoid downloading complete datasets")
    args = p.parse_args()
    count = combine(args.datasets or list(DEFAULT_DATASETS), Path(args.output), args.max_per_dataset, args.seed, args.streaming)
    print(f"wrote {count} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
