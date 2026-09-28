"""固定版本的二十万题图文候选判断数据构建。只下载 train/validation。"""
import argparse
import hashlib
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "race": ("ehovy/race", "2fec9fd81f1dc971569a9b729c43f2f0e6436637", "all"),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "data"),
    "scienceqa": ("derek-thomas/ScienceQA", "f18b0a70359ebfb41f658fd564208d0355b013f4", "data"),
}


def fingerprint(r):
    # 去掉来源与候选顺序，禁止跨来源相同题目进入不同划分。
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().casefold()
    text = norm(r["description"]) + "\n" + "\n".join(sorted(norm(c) for c in r["candidates"]))
    text += "\n" + r.get('image_sha256', '')
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def rows(name, split):
    repo, revision, folder = SOURCES[name]
    suffix = {"train": "1028f23e353fbe3e", "validation": "6c7328ff6c84284c"}
    filename = f"{folder}/{split}-00000-of-00001"
    if name == "scienceqa":
        filename += "-" + suffix[split]
    path = hf_hub_download(repo, filename + ".parquet", repo_type="dataset", revision=revision)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        yield from batch.to_pylist()


def convert(name, split, output):
    for i, r in enumerate(rows(name, split)):
        image_path = None
        image_sha256 = ''
        if name == "race":
            question = f"Passage: {r['article']}\nQuestion: {r['question']}"
            options, gold = r["options"], ord(r["answer"]) - ord("A")
        elif name == "hellaswag":
            question = "Choose the most plausible continuation.\nContext: " + r["ctx"]
            options, gold = r["endings"], int(r["label"])
        else:
            if not r.get("image"):
                continue
            question = r["question"]
            if r.get("hint"):
                question += "\nContext: " + r["hint"]
            # lecture / solution 包含解题信息，不能加入输入。
            options, gold = r["choices"], int(r["answer"])
            image_path = f"images/scienceqa/{split}-{i:06d}.png"
            destination = output / image_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            image = r["image"]
            raw = image.get("bytes")
            with Image.open(io.BytesIO(raw) if raw else image["path"]) as im:
                im.convert("RGB").save(destination)
            image_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        if not 2 <= len(options) <= 26 or not 0 <= gold < len(options):
            raise ValueError((name, split, i))
        yield {"id": f"{name}:{split}:{i}", "description": question.strip(),
               "candidates": [str(c).strip() for c in options], "gold_index": gold,
               "language": "en", "source_dataset": name, "domain": name,
               "source_split": split, "source_id": str(i), "image": image_path,
               "source_revision": SOURCES[name][1], "image_sha256": image_sha256}


def write(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-train", type=Path, default=ROOT / "datasets/selected/train_80k.jsonl")
    p.add_argument("--old-validation", type=Path, default=ROOT / "datasets/selected/validation.jsonl")
    p.add_argument("--exclude", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, default=ROOT / "data/mix200k")
    p.add_argument("--seed", type=int, default=20260926)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    old = read_jsonl(a.old_train)
    validation = read_jsonl(a.old_validation)
    excluded = set(fingerprint(r) for r in validation)
    heldout = ROOT / 'result/evaluation-2026-09-25/heldout_200.jsonl'
    if heldout.exists():
        excluded.update(fingerprint(r) for r in read_jsonl(heldout))
    for path in a.exclude:
        excluded.update(fingerprint(r) for r in read_jsonl(path))
    pools, val_counts = {}, {}
    for name in SOURCES:
        print(f"下载并标准化 {name}", flush=True)
        candidates = list(convert(name, "validation", a.output))
        rng.shuffle(candidates)
        new_val = []
        for r in candidates:
            fp = fingerprint(r)
            if fp not in excluded:
                excluded.add(fp)
                if len(new_val) < 500:
                    new_val.append(r)
        validation.extend(new_val)
        val_counts[name] = len(new_val)
        pools[name] = list(convert(name, "train", a.output))
    seen = set(excluded)
    duplicates = Counter()
    def unique(records, name):
        result = []
        for r in records:
            fp = fingerprint(r)
            if fp in seen:
                duplicates[name] += 1
                continue
            seen.add(fp)
            result.append(r)
        return result
    training = unique(old, "replay80k")
    replay_count = len(training)
    for name in SOURCES:
        rng.shuffle(pools[name])
        pools[name] = unique(pools[name], name)
    # 全部可用含图 ScienceQA，全部 HellaSwag train，RACE 填满总量。
    training.extend(pools["scienceqa"])
    training.extend(pools["hellaswag"])
    race_needed = 200000 - len(training)
    if race_needed < 0 or len(pools["race"]) < race_needed:
        raise ValueError(f"无法满足二十万题配额: RACE 需要 {race_needed}, 可用 {len(pools['race'])}")
    training.extend(pools["race"][:race_needed])
    rng.shuffle(training)
    if len({r["id"] for r in training}) != len(training):
        raise ValueError("重复样本编号")
    if {fingerprint(r) for r in training} & {fingerprint(r) for r in validation}:
        raise ValueError("训练验证交叉")
    write(a.output / "train.jsonl", training)
    write(a.output / "validation.jsonl", validation)
    manifest = {"seed": a.seed, "train_count": len(training), "validation_count": len(validation),
                "replay_count": replay_count, "sources": SOURCES,
                "train_by_source": dict(Counter(r['source_dataset'] for r in training)),
                "train_by_language": dict(Counter(r['language'] for r in training)),
                "image_train_count": sum(bool(r.get('image')) for r in training),
                "new_validation_by_source": val_counts, "duplicates_removed": dict(duplicates),
                "train_validation_overlap": 0,
                "sha256": {n: hashlib.sha256((a.output / n).read_bytes()).hexdigest()
                           for n in ["train.jsonl", "validation.jsonl"]}}
    (a.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
