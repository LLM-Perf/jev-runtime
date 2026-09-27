"""Freeze public engineering fixtures; raw source text stays outside the repository.

Requires httpx and pyarrow. Dataset Viewer split discovery was checked before
freezing these Hub revisions. Published benchmark texts may be in pretraining;
this is not business-domain acceptance or proof of pretraining disjointness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import httpx

SOURCES = {
    "ag_news": {
        "repo": "fancyzhx/ag_news",
        "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        "files": {
            "fit": "data/train-00000-of-00001.parquet",
            "heldout": "data/test-00000-of-00001.parquet",
        },
        "classes": 4,
    },
    "sst2": {
        "repo": "SetFit/sst2",
        "revision": "00ea8ccb7a54b4e3780a3e51aa3f80361ff849c0",
        "files": {"fit": "train.jsonl", "heldout": "test.jsonl"},
        "classes": 2,
    },
}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def select(rows_by_split: dict, per_class: int, classes: int):
    grouped, labels_by_group = {}, {}
    for split, rows in rows_by_split.items():
        grouped[split] = {}
        for index, row in enumerate(rows):
            text, label = row["text"], int(row["label"])
            if not isinstance(text, str) or not text or not 0 <= label < classes:
                raise ValueError("Source row lacks an allowed text/label")
            normalized = unicodedata.normalize("NFKC", " ".join(text.split())).casefold()
            group = sha(normalized.encode())
            labels_by_group.setdefault(group, set()).add(label)
            grouped[split].setdefault(
                group,
                {
                    "sample_id": f"{split}:{index}",
                    "group_id": group,
                    "text": text,
                    "label": label,
                },
            )
    conflicts = {group for group, labels in labels_by_group.items() if len(labels) != 1}
    overlap = set(grouped["fit"]) & set(grouped["heldout"])
    selected = {}
    for split, groups in grouped.items():
        candidates = [
            row
            for group, row in groups.items()
            if group not in conflicts and (split != "fit" or group not in overlap)
        ]
        candidates.sort(key=lambda row: sha(("jev-public-v1:" + row["group_id"]).encode()))
        counts = Counter()
        selected[split] = []
        for row in candidates:
            if counts[row["label"]] < per_class:
                selected[split].append(row)
                counts[row["label"]] += 1
        if dict(counts) != {label: per_class for label in range(classes)}:
            raise ValueError("Insufficient disjoint labeled groups for the frozen balanced sample")
    assert not (
        {r["group_id"] for r in selected["fit"]} & {r["group_id"] for r in selected["heldout"]}
    )
    return selected, {
        "raw_rows": {key: len(value) for key, value in rows_by_split.items()},
        "unique_normalized_groups": {key: len(value) for key, value in grouped.items()},
        "conflicting_groups_excluded": len(conflicts),
        "cross_split_groups_excluded_from_fit": len(overlap),
    }


def run(args):
    if args.output.exists():
        raise ValueError("Keep previous dataset artifacts; choose a new output directory")
    args.output.mkdir(parents=True)
    manifest = {
        "source_commit": args.source_commit,
        "qualification": "public engineering fixtures, not approved business evaluation",
        "selection": "balanced classes; ascending SHA256(jev-public-v1:normalized-group-sha256)",
        "grouping": "NFKC, collapsed whitespace, casefold; exact normalized duplicates only",
        "per_class_per_split": args.per_class,
        "datasets": {},
    }
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for name, source in SOURCES.items():
            directory = args.output / name
            directory.mkdir()
            raw, files = {}, {}
            for split, filename in source["files"].items():
                url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{filename}"
                response = client.get(url)
                response.raise_for_status()
                destination = directory / ("source-" + Path(filename).name)
                destination.write_bytes(response.content)
                files[split] = {
                    "url": url,
                    "sha256": sha(response.content),
                    "bytes": len(response.content),
                }
                if filename.endswith(".parquet"):
                    import pyarrow.parquet as pq

                    raw[split] = pq.read_table(destination, columns=["text", "label"]).to_pylist()
                else:
                    raw[split] = [
                        json.loads(line) for line in response.text.splitlines() if line.strip()
                    ]
            chosen, accounting = select(raw, args.per_class, source["classes"])
            derived = {}
            for split, rows in chosen.items():
                content = "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
                )
                (directory / f"{split}.jsonl").write_text(content)
                derived[split] = {
                    "samples": len(rows),
                    "sha256": sha(content.encode()),
                    "labels": dict(Counter(r["label"] for r in rows)),
                }
            manifest["datasets"][name] = {
                **source,
                "source_files": files,
                "accounting": accounting,
                "derived": derived,
            }
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({"dataset": name, "samples": derived}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--per-class", type=int, default=64)
    args = parser.parse_args()
    if args.per_class <= 0:
        parser.error("per-class must be positive")
    run(args)
