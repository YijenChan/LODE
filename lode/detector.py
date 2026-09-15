"""Process-conditioned GRU event detector used by LODE.

The model predicts the next typed event signature in each responsible process
stream.  It writes chronological event scores and process-seed middleware that
the existing evidence archive and investigator can consume.  Evaluation labels
are intentionally not imported here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_path(config: dict) -> Path:
    return Path(config.get("output_root", ROOT / "runs")) / config["run_id"]


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temp, path)


def save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, value)
    os.replace(temp, path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")


@contextlib.contextmanager
def measured(run: Path, stage: str, **details):
    started = time.perf_counter()
    process = psutil.Process()
    before = process.memory_info().rss
    try:
        yield
    finally:
        append_jsonl(
            run / "metrics" / "stages.jsonl",
            {
                "stage": stage,
                "wall_time_s": time.perf_counter() - started,
                "rss_before_bytes": before,
                "rss_after_bytes": process.memory_info().rss,
                **details,
            },
        )


def relation_vocabulary(config: dict) -> list[str]:
    """Return the fixed event schema declared by the experiment config."""
    relations = list(config["event_schema"]["relations"])
    if len(relations) != len(set(relations)):
        raise ValueError("event_schema.relations contains duplicates")
    return relations


def owner_and_signature(
    src: np.ndarray,
    dst: np.ndarray,
    node_type: np.ndarray,
    relation: np.ndarray,
    clone_relation: int,
    peer_type_count: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return responsible process and (relation, peer type, process role) token."""
    src_type = node_type[src]
    dst_type = node_type[dst]
    owner_is_src = src_type == 0
    owner = np.where(owner_is_src, src, dst).astype(np.int32)
    clone = relation == clone_relation
    owner[clone] = src[clone].astype(np.int32)
    owner_is_src[clone] = True
    if np.any(node_type[owner] != 0):
        raise ValueError("Every retained event must have a responsible process")
    peer_type = np.where(owner_is_src, dst_type, src_type).astype(np.int16)
    role = (~owner_is_src).astype(np.int16)
    signature = ((relation.astype(np.int32) * peer_type_count + peer_type) * 2 + role).astype(np.int16)
    return owner, signature


def prepare(config: dict, split: str) -> None:
    run = run_path(config)
    out = run / "middleware" / split
    complete = out / "metadata.json"
    if complete.exists():
        print(f"reuse prepare-{split}: {complete}")
        return
    input_root = Path(config["data_root"])
    event_path = input_root / "canonical" / split / "sorted" / "events.parquet"
    node_path = input_root / "canonical" / split / "sorted" / "nodes.parquet"
    relations = relation_vocabulary(config)
    entity_types = list(config["event_schema"]["entity_types"])
    if not entity_types or entity_types[0] != "process":
        raise ValueError("event_schema.entity_types must list process first")
    nodes = pq.read_table(node_path, columns=["node_uuid", "node_type"])
    node_ids = nodes["node_uuid"].combine_chunks()
    mapped_types = pc.index_in(nodes["node_type"], value_set=pa.array(entity_types))
    if mapped_types.null_count:
        raise ValueError(f"{split}: node type outside fixed schema")
    node_type = mapped_types.to_numpy().astype(np.int16)
    parquet = pq.ParquetFile(event_path)
    n_events = parquet.metadata.num_rows
    src = np.empty(n_events, dtype=np.int32)
    dst = np.empty(n_events, dtype=np.int32)
    relation = np.empty(n_events, dtype=np.int16)
    timestamp = np.empty(n_events, dtype=np.int64)
    offset = 0
    with measured(run, f"prepare-{split}", events=int(n_events)):
        for batch in parquet.iter_batches(
            columns=["src_uuid", "relation", "dst_uuid", "timestamp_ns"],
            batch_size=524_288,
        ):
            size = len(batch)
            sl = slice(offset, offset + size)
            src_ids = pc.index_in(batch["src_uuid"], value_set=node_ids)
            dst_ids = pc.index_in(batch["dst_uuid"], value_set=node_ids)
            mapped_relations = pc.index_in(batch["relation"], value_set=pa.array(relations))
            if src_ids.null_count or dst_ids.null_count or mapped_relations.null_count:
                raise ValueError(f"{split}: entity or relation outside fixed schema")
            src[sl] = src_ids.to_numpy().astype(np.int32)
            dst[sl] = dst_ids.to_numpy().astype(np.int32)
            relation[sl] = mapped_relations.to_numpy().astype(np.int16)
            timestamp[sl] = batch["timestamp_ns"].to_numpy()
            offset += size
        if offset != n_events or np.any(timestamp[1:] < timestamp[:-1]):
            raise ValueError(f"{split}: canonical alignment or chronology failed")
        owner, signature = owner_and_signature(
            src, dst, node_type, relation, relations.index("EVENT_CLONE"), len(entity_types)
        )
        order = np.argsort(owner, kind="stable")
        counts = np.bincount(owner, minlength=len(node_type)).astype(np.int64)
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        save_npy(out / "owner.npy", owner)
        save_npy(out / "signature.npy", signature)
        save_npy(out / "timestamp_ns.npy", timestamp)
        save_npy(out / "sorted_signature.npy", signature[order])
        save_npy(out / "sorted_event_pos.npy", order.astype(np.int32))
        save_npy(out / "owner_offsets.npy", offsets)
        metadata = {
            "split": split,
            "events": int(n_events),
            "nodes": int(len(node_type)),
            "active_processes": int(np.count_nonzero(counts)),
            "relations": relations,
            "entity_types": entity_types,
            "signature": "(relation, peer_entity_type, responsible_process_role)",
            "signature_vocabulary": int(len(relations) * len(entity_types) * 2),
            "owner_rule": "process endpoint; source process for CLONE",
            "chronological_within_process": True,
            "labels_used": False,
        }
        save_json(complete, metadata)
    print(json.dumps(metadata))


class EventGRU(nn.Module):
    def __init__(self, vocabulary: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocabulary)


def owner_batches(offsets: np.ndarray, batch_size: int) -> list[np.ndarray]:
    owners = np.flatnonzero(np.diff(offsets) > 0)
    lengths = np.diff(offsets)[owners]
    owners = owners[np.argsort(-lengths, kind="stable")]
    return [owners[i : i + batch_size] for i in range(0, len(owners), batch_size)]


def chunk(
    signatures: np.ndarray,
    offsets: np.ndarray,
    owners: np.ndarray,
    start: int,
    width: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    lengths = np.minimum(np.maximum(np.diff(offsets)[owners] - start, 0), width).astype(np.int64)
    maximum = int(lengths.max(initial=0))
    if maximum == 0:
        return None, lengths
    values = np.zeros((len(owners), maximum), dtype=np.int64)
    for row, (owner, length) in enumerate(zip(owners, lengths)):
        if length:
            begin = offsets[owner] + start
            values[row, :length] = signatures[begin : begin + length]
    return values, lengths


def next_event_loss(
    model: EventGRU,
    target: torch.Tensor,
    lengths: np.ndarray,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    device = target.device
    active = torch.as_tensor(lengths > 0, device=device)
    logits0 = model.output(hidden.squeeze(0))
    total = nn.functional.cross_entropy(logits0[active], target[active, 0], reduction="sum")
    count = int(active.sum())
    output, next_hidden = model.gru(model.embedding(target), hidden)
    if target.shape[1] > 1:
        positions = torch.arange(target.shape[1] - 1, device=device)[None, :]
        mask = positions < torch.as_tensor(lengths - 1, device=device)[:, None]
        expected = target[:, 1:][mask]
        if len(expected):
            logits = model.output(output[:, :-1])[mask]
            total = total + nn.functional.cross_entropy(logits, expected, reduction="sum")
            count += int(len(expected))
    return total / max(count, 1), next_hidden, count


def train(config: dict) -> None:
    run = run_path(config)
    checkpoint_path = run / "native" / "gru_detector.pt"
    if checkpoint_path.exists():
        print(f"reuse train: {checkpoint_path}")
        return
    source = run / "middleware" / "train"
    signatures = np.load(source / "sorted_signature.npy", mmap_mode="r")
    offsets = np.load(source / "owner_offsets.npy", mmap_mode="r")
    pids = config["pids"]
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = int(json.loads((source / "metadata.json").read_text())["signature_vocabulary"])
    model = EventGRU(vocabulary, int(pids["embedding_dim"]), int(pids["hidden_dim"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(pids["learning_rate"]))
    batches = owner_batches(offsets, int(pids["batch_processes"]))
    history = []
    with measured(run, "train-gru", device=str(device), epochs=int(pids["epochs"])):
        for epoch in range(int(pids["epochs"])):
            model.train()
            epoch_started = time.perf_counter()
            total_loss = 0.0
            total_events = 0
            for owners in batches:
                longest = int(np.diff(offsets)[owners].max())
                hidden = torch.zeros(1, len(owners), int(pids["hidden_dim"]), device=device)
                for start in range(0, longest, int(pids["tbptt_steps"])):
                    values, lengths = chunk(
                        signatures, offsets, owners, start, int(pids["tbptt_steps"])
                    )
                    if values is None:
                        break
                    target = torch.from_numpy(values).to(device)
                    loss, hidden, count = next_event_loss(model, target, lengths, hidden)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    hidden = hidden.detach()
                    total_loss += float(loss.detach()) * count
                    total_events += count
            row = {
                "epoch": epoch + 1,
                "mean_nll": total_loss / total_events,
                "events": total_events,
                "wall_time_s": time.perf_counter() - epoch_started,
            }
            history.append(row)
            append_jsonl(run / "native" / "train_history.jsonl", row)
            print(json.dumps(row), flush=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.cpu().state_dict(),
                "vocabulary": vocabulary,
                "embedding_dim": int(pids["embedding_dim"]),
                "hidden_dim": int(pids["hidden_dim"]),
                "history": history,
                "seed": seed,
                "semantics": "predict signature before updating the responsible process state",
            },
            checkpoint_path,
        )


@torch.no_grad()
def score(config: dict, split: str) -> None:
    run = run_path(config)
    target_path = run / "middleware" / f"event_scores_{split}.npy"
    if target_path.exists():
        print(f"reuse score-{split}: {target_path}")
        return
    source = run / "middleware" / split
    signatures = np.load(source / "sorted_signature.npy", mmap_mode="r")
    event_pos = np.load(source / "sorted_event_pos.npy", mmap_mode="r")
    offsets = np.load(source / "owner_offsets.npy", mmap_mode="r")
    checkpoint = torch.load(run / "native" / "gru_detector.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventGRU(
        checkpoint["vocabulary"], checkpoint["embedding_dim"], checkpoint["hidden_dim"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    sorted_scores = np.empty(len(signatures), dtype=np.float32)
    width = int(config["pids"]["tbptt_steps"])
    with measured(run, f"score-{split}", device=str(device), events=len(signatures)):
        for owners in owner_batches(offsets, int(config["pids"]["batch_processes"])):
            longest = int(np.diff(offsets)[owners].max())
            hidden = torch.zeros(1, len(owners), checkpoint["hidden_dim"], device=device)
            for start in range(0, longest, width):
                values, lengths = chunk(signatures, offsets, owners, start, width)
                if values is None:
                    break
                target = torch.from_numpy(values).to(device)
                logits0 = model.output(hidden.squeeze(0))
                first = -nn.functional.log_softmax(logits0, -1).gather(
                    1, target[:, :1]
                ).squeeze(1)
                output, hidden = model.gru(model.embedding(target), hidden)
                rest = None
                if target.shape[1] > 1:
                    rest = -nn.functional.log_softmax(model.output(output[:, :-1]), -1).gather(
                        2, target[:, 1:, None]
                    ).squeeze(2)
                for row, (owner, length) in enumerate(zip(owners, lengths)):
                    if not length:
                        continue
                    begin = offsets[owner] + start
                    sorted_scores[begin] = float(first[row])
                    if length > 1:
                        sorted_scores[begin + 1 : begin + length] = (
                            rest[row, : length - 1].cpu().numpy()
                        )
                hidden = hidden.detach()
        chronological = np.empty_like(sorted_scores)
        chronological[event_pos] = sorted_scores
        save_npy(target_path, chronological)
        save_json(
            run / "middleware" / f"score_{split}.json",
            {
                "split": split,
                "events": len(chronological),
                "mean_nll": float(chronological.mean()),
                "finite": bool(np.isfinite(chronological).all()),
                "labels_used": False,
            },
        )
    print(json.dumps({"split": split, "events": len(chronological)}))


def process_windows(
    event_pos: np.ndarray,
    offsets: np.ndarray,
    timestamp: np.ndarray,
    scores: np.ndarray,
    window_ns: int,
) -> tuple[np.ndarray, list[dict]]:
    maxima = []
    records = []
    for owner in np.flatnonzero(np.diff(offsets) > 0):
        positions = np.asarray(event_pos[offsets[owner] : offsets[owner + 1]], dtype=np.int64)
        windows = timestamp[positions] // window_ns
        starts = np.r_[0, np.flatnonzero(windows[1:] != windows[:-1]) + 1]
        values = np.maximum.reduceat(scores[positions], starts)
        maxima.extend(values.tolist())
        best_local = int(np.argmax(scores[positions]))
        anchor = int(positions[best_local])
        records.append(
            {
                "owner": int(owner),
                "anchor": anchor,
                "window": int(timestamp[anchor] // window_ns),
                "score": float(scores[anchor]),
            }
        )
    return np.asarray(maxima, dtype=np.float32), records


def calibrate(config: dict) -> None:
    run = run_path(config)
    target = run / "middleware" / "all_seeds.json"
    if target.exists():
        print(f"reuse calibrate: {target}")
        return
    window_ns = int(config["pids"]["window_seconds"]) * 1_000_000_000
    split_values = {}
    split_records = {}
    with measured(run, "calibrate-process-seeds"):
        for split in ("validation", "test"):
            source = run / "middleware" / split
            values, records = process_windows(
                np.load(source / "sorted_event_pos.npy", mmap_mode="r"),
                np.load(source / "owner_offsets.npy", mmap_mode="r"),
                np.load(source / "timestamp_ns.npy", mmap_mode="r"),
                np.load(run / "middleware" / f"event_scores_{split}.npy", mmap_mode="r"),
                window_ns,
            )
            split_values[split] = values
            split_records[split] = records
        reference = np.sort(split_values["validation"])
        if not len(reference):
            raise ValueError("No active benign validation process windows")
        eta = float(config["pids"]["alert_percentile"])
        seeds = []
        for row in split_records["test"]:
            q = float(np.searchsorted(reference, row["score"], side="right") / len(reference))
            if q >= eta:
                seeds.append({**row, "q": q})
        seeds.sort(key=lambda row: (row["anchor"], row["owner"]))
        save_json(target, seeds)
        save_npy(run / "middleware" / "validation_process_window_scores.npy", reference)
        save_json(
            run / "middleware" / "calibration.json",
            {
                "reference": "active benign validation process-window maxima",
                "aggregation": "maximum event NLL per process window; maximum window per process",
                "quantile": "right empirical CDF",
                "eta": eta,
                "validation_process_windows": int(len(reference)),
                "test_active_processes": int(len(split_records["test"])),
                "seeds": int(len(seeds)),
                "labels_used": False,
            },
        )
    print(json.dumps({"seeds": len(seeds), "eta": eta}))


def handoff(config: dict) -> None:
    run = run_path(config)
    input_root = Path(config["data_root"])
    events = pq.ParquetFile(
        input_root / "canonical" / "test" / "sorted" / "events.parquet"
    ).metadata.num_rows
    scores = np.load(run / "middleware" / "event_scores_test.npy", mmap_mode="r")
    seeds = json.loads((run / "middleware" / "all_seeds.json").read_text())
    if len(scores) != events or any(not 0 <= row["anchor"] < events for row in seeds):
        raise ValueError("Detector-to-archive event-position contract failed")
    archive_meta = Path(config.get("archive_root", run / "archive")) / "complete.json"
    archive_events = None
    if archive_meta.exists():
        archive_events = json.loads(archive_meta.read_text())["events"]
        if archive_events != events:
            raise ValueError("Existing evidence archive is not aligned with detector scores")
    save_json(
        run / "middleware" / "handoff.json",
        {
            "status": "completed_pipeline_only",
            "event_scores": str(run / "middleware" / "event_scores_test.npy"),
            "seed_file": str(run / "middleware" / "all_seeds.json"),
            "test_events": int(events),
            "seed_count": int(len(seeds)),
            "archive_event_count": archive_events,
            "compatible_consumers": ["lode.archive", "lode.investigator", "lode.commit"],
            "labels_used": False,
        },
    )
    print(json.dumps({"handoff": "passed", "events": events, "seeds": len(seeds)}))


def initialize(config_path: Path, config: dict) -> None:
    run = run_path(config)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config").mkdir(exist_ok=True)
    resolved = run / "config" / "resolved.yaml"
    if not resolved.exists():
        shutil.copy2(config_path, resolved)
    code_snapshot = run / "code-snapshots" / "detector.py"
    code_snapshot.parent.mkdir(exist_ok=True)
    if not code_snapshot.exists():
        shutil.copy2(Path(__file__), code_snapshot)
    protocol = Path(config["protocol"])
    if not protocol.exists():
        raise FileNotFoundError(f"trajectory protocol not found: {protocol}")
    save_json(
        run / "run.json",
        {
            "run_id": config["run_id"],
            "status": "running_pipeline_only",
            "scope": "LODE process-conditioned detector run",
            "seed": config["seed"],
            "interpreter": os.sys.executable,
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "test_labels_used": False,
        },
    )


def finish(config: dict) -> None:
    run = run_path(config)
    row = json.loads((run / "run.json").read_text())
    row["status"] = "completed_pipeline_only"
    row["completed_at_unix_s"] = time.time()
    row["outputs"] = [
        "native/gru_detector.pt",
        "middleware/event_scores_validation.npy",
        "middleware/event_scores_test.npy",
        "middleware/all_seeds.json",
        "middleware/handoff.json",
    ]
    save_json(run / "run.json", row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "command",
        choices=["prepare", "train", "score", "calibrate", "handoff", "all"],
    )
    parser.add_argument("--split", choices=["train", "validation", "test"])
    args = parser.parse_args()
    config = load_config(args.config)
    initialize(args.config, config)
    if args.command == "prepare":
        if not args.split:
            parser.error("prepare requires --split")
        prepare(config, args.split)
    elif args.command == "train":
        train(config)
    elif args.command == "score":
        if args.split not in ("validation", "test"):
            parser.error("score requires --split validation|test")
        score(config, args.split)
    elif args.command == "calibrate":
        calibrate(config)
    elif args.command == "handoff":
        handoff(config)
    else:
        for split in ("train", "validation", "test"):
            prepare(config, split)
        train(config)
        for split in ("validation", "test"):
            score(config, split)
        calibrate(config)
        handoff(config)
        finish(config)


if __name__ == "__main__":
    main()
