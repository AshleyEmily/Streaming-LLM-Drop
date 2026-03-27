"""
Quick inspection of the wikitext-2-raw-v1 test split.
Run this on the cluster to verify the dataset structure before eval.

Usage:
    python scripts/inspect_dataset.py
"""
from datasets import load_dataset

data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

print(f"Total entries : {len(data)}")
print(f"Columns       : {data.column_names}")
print()
print("First 10 entries:")
for i, text in enumerate(data["text"][:10]):
    print(f"  [{i}] chars={len(text):<6}  repr={repr(text[:80])}")

print()
non_empty = [t for t in data["text"] if t.strip()]
total_chars = sum(len(t) for t in data["text"])
print(f"Non-empty entries : {len(non_empty)} / {len(data)}")
print(f"Total chars       : {total_chars:,}")
print(f"Avg chars/entry   : {total_chars / len(data):.1f}")
