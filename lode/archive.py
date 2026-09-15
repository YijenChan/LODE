"""Disk-backed entity/time postings for LODE's retain-all evidence space."""
from __future__ import annotations

import argparse
import bisect
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


def _save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build(config: dict) -> Path:
    data_root = Path(config["data_root"])
    archive_root = Path(config.get("archive_root", Path(config["output_root"]) / "archive"))
    split = config.get("investigation_split", "test")
    source = data_root / "canonical" / split / "sorted" / "events.parquet"
    node_path = data_root / "canonical" / split / "sorted" / "nodes.parquet"
    archive_root.mkdir(parents=True, exist_ok=True)
    complete = archive_root / "complete.json"
    if complete.exists():
        return archive_root

    event_file = pq.ParquetFile(source)
    n_events = event_file.metadata.num_rows
    node_ids = pq.read_table(node_path, columns=["node_uuid"])["node_uuid"].combine_chunks()
    relations = list(config["event_schema"]["relations"])
    arrays = {
        name: np.lib.format.open_memmap(archive_root / f"{name}.npy", mode="w+", dtype=dtype, shape=(n_events,))
        for name, dtype in (("src", "int32"), ("dst", "int32"), ("rel", "int16"), ("ts", "int64"))
    }
    offset = 0
    started = time.perf_counter()
    for batch in event_file.iter_batches(
        columns=["src_uuid", "dst_uuid", "relation", "timestamp_ns"], batch_size=262_144
    ):
        size = len(batch)
        sl = slice(offset, offset + size)
        for name, column, vocabulary in (
            ("src", "src_uuid", node_ids), ("dst", "dst_uuid", node_ids),
            ("rel", "relation", pa.array(relations)),
        ):
            mapped = pc.index_in(batch[column], value_set=vocabulary)
            if mapped.null_count:
                raise ValueError(f"unmapped value in {column}")
            arrays[name][sl] = mapped.to_numpy()
        arrays["ts"][sl] = batch["timestamp_ns"].to_numpy()
        offset += size
    if offset != n_events or np.any(arrays["ts"][1:] < arrays["ts"][:-1]):
        raise ValueError("canonical events must be aligned and chronological")
    for array in arrays.values():
        array.flush()

    for endpoint in ("src", "dst"):
        order = np.argsort(arrays[endpoint], kind="stable").astype(np.int32)
        offsets = np.r_[0, np.cumsum(np.bincount(arrays[endpoint], minlength=len(node_ids)), dtype=np.int64)]
        np.save(archive_root / f"{endpoint}_postings.npy", order)
        np.save(archive_root / f"{endpoint}_offsets.npy", offsets)

    row_groups = np.cumsum(
        [0] + [event_file.metadata.row_group(i).num_rows for i in range(event_file.num_row_groups)]
    ).tolist()
    _save(complete, {
        "events": n_events, "nodes": len(node_ids), "relations": relations,
        "source": str(source.resolve()), "nodes_source": str(node_path.resolve()),
        "row_groups": row_groups,
        "extra_index_bytes": sum(path.stat().st_size for path in archive_root.glob("*.npy")),
        "build_wall_time_s": time.perf_counter() - started,
        "construction": "numeric source/destination postings; no provenance graph materialized",
    })
    return archive_root


class Archive:
    def __init__(self, archive_root: Path):
        self.root = Path(archive_root)
        self.meta = json.loads((self.root / "complete.json").read_text(encoding="utf-8"))
        self.rels = self.meta["relations"]
        self.a = {
            name: np.load(self.root / f"{name}.npy", mmap_mode="r")
            for name in ("src", "dst", "rel", "ts", "src_postings", "src_offsets", "dst_postings", "dst_offsets")
        }
        self.nodes = pq.read_table(self.meta["nodes_source"]).to_pylist()
        self.raw = pq.ParquetFile(self.meta["source"])
        self.raw_cache: dict[int, object] = {}

    def close(self) -> None:
        """Release memory-mapped postings, which is required on Windows."""
        self.raw_cache.clear()
        close_raw = getattr(self.raw, "close", None)
        if close_raw is not None:
            close_raw()
        for array in self.a.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def event(self, position: int) -> dict:
        p = int(position)
        return {"id": p, "src": int(self.a["src"][p]), "dst": int(self.a["dst"][p]),
                "rel": self.rels[int(self.a["rel"][p])], "ts": int(self.a["ts"][p])}

    def attributes(self, node: int) -> str:
        row = self.nodes[int(node)]
        attrs = json.loads(row["attributes_json"])
        value = attrs.get("path") or attrs.get("filename")
        if not value:
            value = f"{attrs.get('remoteAddress', '')}:{attrs.get('remotePort', '')}".strip(":")
        return str(value or row["node_type"])[:120]

    def incident(self, node: int, anchor: int, side: str = "past", order: str = "near",
                 limit: int = 24, cursor: int | None = None) -> tuple[list[dict], dict]:
        """Return a bounded cursor-paged packet from both endpoint postings."""
        if side not in {"past", "future"} or order not in {"near", "earliest"}:
            raise ValueError("unsupported retrieval operator")
        started = time.perf_counter()
        node, anchor = int(node), int(anchor)
        posts: list[int] = []
        available = 0
        ascending = order == "earliest" or side == "future"
        for endpoint in ("src", "dst"):
            offsets = self.a[f"{endpoint}_offsets"]
            values = self.a[f"{endpoint}_postings"][offsets[node]:offsets[node + 1]]
            cut = np.searchsorted(values, anchor, side="left" if side == "past" else "right")
            values = values[:cut] if side == "past" else values[cut:]
            if cursor is not None:
                index = np.searchsorted(values, cursor, side="right" if ascending else "left")
                values = values[index:] if ascending else values[:index]
            available += len(values)
            values = values[:4096] if ascending else values[-4096:][::-1]
            posts.extend(values.tolist())
        posts = sorted(set(posts), reverse=not ascending)[:4096]

        selected: list[dict] = []
        seen: set[tuple[int, str, int]] = set()
        last = None
        for position in posts:
            last = position
            event = self.event(position)
            peer = event["dst"] if event["src"] == node else event["src"]
            key = (peer, event["rel"], event["ts"] // 900_000_000_000)
            if key in seen:
                continue
            seen.add(key)
            selected.append(event)
            if len(selected) >= limit:
                break
        return selected, {
            "node": node, "anchor": anchor, "side": side, "order": order,
            "rows_examined": len(posts), "capped_scan": available > 4096,
            "returned": len(selected), "cursor": last,
            "latency_s": time.perf_counter() - started,
        }

    def raw_event(self, position: int) -> dict:
        group = bisect.bisect_right(self.meta["row_groups"], int(position)) - 1
        if group not in self.raw_cache:
            self.raw_cache = {group: self.raw.read_row_group(
                group, columns=["event_id", "source_file", "source_line"]
            )}
        offset = int(position) - self.meta["row_groups"][group]
        return self.raw_cache[group].slice(offset, 1).to_pylist()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(build(config))


if __name__ == "__main__":
    main()
