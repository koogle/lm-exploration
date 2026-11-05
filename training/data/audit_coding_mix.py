"""Bounded real-source preparation smoke test, using the production GLM shard path.

This produces a local, sampled corpus, not a full-corpus readiness certificate.
Run from training/: python -m training_data.audit_coding_mix --plan <json> --out <dir>
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from training_data import packed_shards as packed
from training_data.licenses import PERMISSIVE_LICENSES
from training_data.packed_shards import document_key, partition

from tokenizer import glm


def select_rows(rows, source, seen):
    splits = [[], []]
    dropped = Counter()
    for index, row in enumerate(rows):
        text = row.get(source["column"])
        if not isinstance(text, str) or not text.strip():
            dropped["empty"] += 1
            continue
        if source.get("permissive_licenses_only") and (
            str(row.get("license", "")).lower() not in PERMISSIVE_LICENSES
        ):
            dropped["license"] += 1
            continue
        key = document_key(text)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        split_key = key
        if source.get("split_column"):
            group = row.get(source["split_column"])
            if not group:
                dropped["missing_split_group"] += 1
                continue
            split_key = document_key(str(group).lower())
        splits[partition(split_key)].append(
            {
                "text": text,
                "document_sha256": key,
                "source_row": row.get("_source_row", index),
                "source_repo": source["repo"],
                "source_revision": source["revision"],
                "source_file": row.get("_source_file", source.get("file", "")),
                "split_group": row.get(source.get("split_column", ""), ""),
                "language": str(row.get("language", "")),
                "license": str(row.get("license", "")),
            }
        )
    return splits, dict(dropped)


def sample_parquet(parquet, source, limit, seed):
    """Seeded rows across selected row groups; preserve upstream row positions."""
    rng = np.random.default_rng(seed)
    columns = [source["column"]]
    columns += [
        c
        for c in ("license", "language", source.get("split_column"))
        if c and c in parquet.schema_arrow.names and c not in columns
    ]
    groups = np.arange(parquet.num_row_groups)
    if source.get("sample_row_groups"):
        groups = np.sort(
            rng.choice(groups, min(len(groups), source["sample_row_groups"]), replace=False)
        )
    total = sum(parquet.metadata.row_group(int(g)).num_rows for g in groups)
    selected = np.sort(rng.choice(total, min(limit, total), replace=False))
    rows, offset = [], 0
    for group in groups:
        upstream = sum(parquet.metadata.row_group(g).num_rows for g in range(int(group)))
        consumed = 0
        for batch in parquet.iter_batches(
            batch_size=1024, row_groups=[int(group)], columns=columns
        ):
            take = selected[(selected >= offset) & (selected < offset + len(batch))] - offset
            if len(take):
                for row, index in zip(batch.take(pa.array(take)).to_pylist(), take, strict=True):
                    rows.append({**row, "_source_row": upstream + consumed + int(index)})
            offset += len(batch)
            consumed += len(batch)
    return rows


def remote_rows(source, limit):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = source.get("sample_files", [source.get("file")])
    rows = []
    for index, file in enumerate(files):
        path = f"datasets/{source['repo']}@{source['revision']}/{file}"
        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            if "sample_files" in source:
                count = limit // len(files) + (index < limit % len(files))
                found = sample_parquet(
                    parquet, source, count, source.get("sample_seed", 1234) + index
                )
            else:
                columns = [source["column"]]
                if "license" in parquet.schema_arrow.names:
                    columns.append("license")
                found = []
                for batch in parquet.iter_batches(batch_size=min(limit, 256), columns=columns):
                    found.extend(batch.to_pylist()[: limit - len(found)])
                    if len(found) >= limit:
                        break
            rows.extend({**row, "_source_file": file} for row in found)
        print(f"Sampled {file}: {len(found)} rows", flush=True)
    return rows


def check_prepared(recipe, root):
    counts = {}
    for name, source in recipe["sources"].items():
        counts[name] = []
        for index in range(source["shards"]):
            binary, docs, metadata = packed.shard_paths(root, name, index)
            tokens = np.fromfile(binary, np.uint32)
            lengths = np.load(docs)
            meta = json.loads(metadata.read_text())
            assert len(tokens) == int(lengths[:, 0].sum()) == meta["tokens"]
            assert len(lengths) > 0, f"{name}: empty split after filtering"
            assert int(tokens.max()) < glm.TOKENIZER_SIZE
            assert np.all(tokens[np.cumsum(lengths[:, 0]) - 1] == glm.EOS_ID)
            assert np.all(lengths[:, 1] > 0)
            counts[name].append(meta["tokens"])
    return counts


def check_model(config_path, recipe, root):
    """Two CPU updates on a short real batch; no claim of model quality or GPU readiness."""
    import torch
    from model_hybrid.documents import mtp_labels
    from model_hybrid.model import HybridConfig, HybridLM

    torch.set_num_threads(4)
    torch.manual_seed(1234)
    config = json.loads(config_path.read_text())
    model = HybridLM(HybridConfig(**config["model"]))
    # Short sequences keep the CPU reference recurrence inexpensive. The tokenizer,
    # architecture, document masking and objective are the pilot's actual ones.
    mixture = packed.PreparedMixture(recipe, root, wait_seconds=0)
    x, y, names = mixture.next_batch(2)
    x, y = torch.from_numpy(x[:, :16]).long(), torch.from_numpy(y[:, :16]).long()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses, norms = [], []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = model(x, y, mtp_labels=mtp_labels(x, y, glm.EOS_ID))
        loss = result["loss"] / result["count"]
        loss = loss + 0.1 * result["mtp_loss"] / result["mtp_count"].clamp_min(1)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite CPU model loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        losses.append(loss.item())
        norms.append(norm.item())
        optimizer.step()
    if losses[-1] >= losses[0]:
        raise ValueError("Fixed-minibatch CPU smoke did not reduce loss")
    return {
        "scope": "two CPU SGD steps, fixed 2x16-token batch; not a training experiment",
        "parameters": sum(p.numel() for p in model.parameters()),
        "sources": names,
        "losses": losses,
        "gradient_norms": norms,
    }


def check_reasoning_source(source, tokenizer, screen, ngram):
    """Probe the existing mid-training renderer with complete real reasoning rows."""
    from training_data.chat import RENDERERS

    rows = remote_rows(source, 16)
    lengths, screened = [], 0
    for row in rows:
        text = RENDERERS["glm_chat_thinking"](row[source["column"]])
        if "<think>" not in text or text.count("<think>") != text.count("</think>"):
            raise ValueError("Reasoning probe has missing or unbalanced thinking spans")
        if not text.rsplit("</think>", 1)[1].strip():
            raise ValueError("Reasoning probe has no answer after the thinking span")
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded.count("<think>") != text.count("<think>"):
            raise ValueError("Thinking tokens lost in tokenization")
        if decoded.count("</think>") != text.count("</think>"):
            raise ValueError("Closing thinking tokens lost in tokenization")
        lengths.append(len(ids) + 1)  # EOS appended by shard preparation
        screened += int(packed.contaminated(ids, screen, ngram))
    return {
        "upstream": source,
        "rows": len(rows),
        "tokens": lengths,
        "complete_rows_fitting_context": {
            str(size): sum(length <= size for length in lengths) for size in (4096, 8192, 32768)
        },
        "screened_rows": screened,
        "reasoning_and_answer_preserved": True,
    }


def audit(plan_path, out, rows_per_source, windows, model_config=None, resume=False):
    from training_data.subword import digest

    plan = json.loads(plan_path.read_text())
    pending = [
        name for name, source in plan["sources"].items() if source.get("access_status") == "pending"
    ]
    if pending:
        raise ValueError(f"Source access pending: {', '.join(pending)}; keep original weights")
    if abs(sum(s["weight"] for s in plan["sources"].values()) - 1) > 1e-9:
        raise ValueError("Weights must sum to one")
    if rows_per_source < 100 or windows < 20:
        raise ValueError("Use at least 100 rows/source and 20 mixture windows")
    out.mkdir(parents=True, exist_ok=resume)
    if resume and (out / "plan.json").read_bytes() != plan_path.read_bytes():
        raise ValueError("Resume requires identical sample plan")
    (out / "plan.json").write_bytes(plan_path.read_bytes())
    report = {
        "status": "incomplete",
        "scope": "bounded source sample; sampling method is recorded per source; near-dedup pending",
        "plan_sha256": digest(plan_path),
        "rows_per_source": rows_per_source,
        "sources": {},
    }
    report_path = out / "audit.json"
    if resume:
        report = json.loads(report_path.read_text())
        report["status"] = "incomplete"
    report_path.write_text(json.dumps(report, indent=2))
    try:
        recipe = copy.deepcopy(plan)
        recipe["name"] = out.name
        recipe["sources"] = {}
        seen = set()
        local_files = {}
        for name, source in plan["sources"].items():
            if resume and name in report["sources"]:
                cached = report["sources"][name]
                files = [f"{name}-{i}.parquet" for i in range(2)]
                for file, expected in zip(files, cached["raw_sha256"], strict=True):
                    path = out / "raw" / file
                    if digest(path) != expected:
                        raise ValueError(f"Cached sample checksum mismatch: {file}")
                    seen.update(
                        pq.read_table(path, columns=["document_sha256"]).column(0).to_pylist()
                    )
                    local_files[file] = path
                recipe["sources"][name] = {
                    "weight": source["weight"],
                    "repo": "local-audit-samples",
                    "revision": report["plan_sha256"],
                    "files": files,
                    "shards": 2,
                    "column": "text",
                    "max_passes": 1,
                }
                print(f"Reusing verified sample {name}", flush=True)
                continue
            print(f"Sampling {name}", flush=True)
            rows = remote_rows(source, source.get("sample_rows", rows_per_source))
            splits, dropped = select_rows(rows, source, seen)
            files = []
            for index, split in enumerate(splits):
                if not split:
                    raise ValueError(f"{name}: empty split {index}; increase sample size")
                file = f"{name}-{index}.parquet"
                path = out / "raw" / file
                path.parent.mkdir(exist_ok=True)
                pq.write_table(pa.Table.from_pylist(split), path, row_group_size=1024)
                local_files[file] = path
                files.append(file)
            recipe["sources"][name] = {
                "weight": source["weight"],
                "repo": "local-audit-samples",
                "revision": report["plan_sha256"],
                "files": files,
                "shards": 2,
                "column": "text",
                "max_passes": 1,
            }
            report["sources"][name] = {
                "upstream": source,
                "sampled": len(rows),
                "split_documents": list(map(len, splits)),
                "dropped": dropped,
                "raw_sha256": [digest(local_files[f]) for f in files],
                "languages": dict(Counter(str(r.get("language", "unknown")) for r in rows)),
            }
            report_path.write_text(json.dumps(report, indent=2))
        print("Building benchmark screen", flush=True)
        tokenizer = glm.load()
        screen, screening = packed.build_screen(plan, tokenizer)
        np.save(out / "screen.npy", screen)
        report["screen"] = screening
        if plan.get("reasoning_probe"):
            report["reasoning_probe"] = check_reasoning_source(
                plan["reasoning_probe"], tokenizer, screen, plan["contamination"]["ngram"]
            )
        root = out / "prepared"

        def download(repo, revision, file):
            if repo != "local-audit-samples" or revision != report["plan_sha256"]:
                raise ValueError("Unexpected local sample identity")
            return local_files[file]

        for name in recipe["sources"]:
            for index in range(2):
                packed.prepare_shard(
                    recipe, name, index, root, tokenizer, screen, glm.EOS_ID, download
                )
        report["prepared_tokens"] = check_prepared(recipe, root)
        mixture = packed.PreparedMixture(recipe, root, wait_seconds=0)
        x, y, names = mixture.next_batch(windows)
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
        observed = Counter(names)
        for name, source in recipe["sources"].items():
            assert abs(observed[name] - windows * source["weight"]) <= 1
        state = json.loads(json.dumps(mixture.state_dict()))
        expected = mixture.next_batch(4)
        resumed = packed.PreparedMixture(recipe, root, wait_seconds=0)
        resumed.load_state_dict(state)
        actual = resumed.next_batch(4)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[2] == expected[2]
        # Preserve reasoning markers as distinct IDs through tokenization and decoding.
        trace = "<think>Check the boundary case.</think>\nreturn 0"
        ids = tokenizer.encode(trace, add_special_tokens=False).ids
        assert glm.SPECIAL_TOKENS["<think>"] in ids
        assert glm.SPECIAL_TOKENS["</think>"] in ids
        assert tokenizer.decode(ids, skip_special_tokens=False) == trace
        report.update(
            status="passed",
            mixture_windows=dict(observed),
            windows=windows,
            resume_exact=True,
            reasoning_tokens_preserved=True,
        )
        # A conservative single-pass target leaves room for rounding and boundary tokens.
        recipe["target_tokens"] = int(
            0.9
            * min(
                report["prepared_tokens"][n][0] / s["weight"] for n, s in recipe["sources"].items()
            )
        )
        if model_config:
            print("Checking small model on CPU", flush=True)
            report["model_smoke"] = check_model(model_config, recipe, root)
        (out / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-per-source", type=int, default=512)
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.plan, args.out, args.rows_per_source, args.windows, args.model_config, args.resume
    )
    print(json.dumps({"status": result["status"], "report": str(args.out / "audit.json")}))


if __name__ == "__main__":
    main()
