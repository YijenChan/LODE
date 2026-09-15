import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lode.archive import Archive, build


class ArchiveTests(unittest.TestCase):
    def fixture(self, n=20):
        archive = Archive.__new__(Archive)
        archive.rels = ["EVENT_READ"]
        src = np.zeros(n, dtype=np.int32)
        dst = np.arange(1, n + 1, dtype=np.int32)
        archive.a = {
            "src": src, "dst": dst, "rel": np.zeros(n, dtype=np.int16),
            "ts": np.arange(n, dtype=np.int64) * 1_000_000_000,
            "src_postings": np.arange(n), "src_offsets": np.r_[0, n, np.repeat(n, n - 1)],
            "dst_postings": np.arange(n), "dst_offsets": np.r_[0, 0, np.arange(1, n + 1)],
        }
        return archive

    def test_earliest_and_near(self):
        archive = self.fixture()
        earliest, _ = archive.incident(0, 15, "past", "earliest", 2)
        near, _ = archive.incident(0, 15, "past", "near", 2)
        self.assertEqual([event["id"] for event in earliest], [0, 1])
        self.assertEqual([event["id"] for event in near], [14, 13])

    def test_future_is_anchor_exclusive(self):
        archive = self.fixture()
        packet, _ = archive.incident(0, 15, "future", "near", 2)
        self.assertEqual([event["id"] for event in packet], [16, 17])

    def test_build_preserves_raw_reference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "canonical" / "test" / "sorted"
            split.mkdir(parents=True)
            pq.write_table(pa.table({
                "node_uuid": ["p", "f"], "node_type": ["process", "file"],
                "attributes_json": ["{}", "{\"filename\":\"/tmp/x\"}"],
                "source_file": ["nodes", "nodes"], "source_line": [1, 2],
            }), split / "nodes.parquet")
            pq.write_table(pa.table({
                "event_id": ["event-1"], "timestamp_ns": [1],
                "src_uuid": ["p"], "dst_uuid": ["f"], "relation": ["EVENT_WRITE"],
                "source_file": ["audit.log"], "source_line": [7],
            }), split / "events.parquet")
            archive_root = root / "archive"
            build({"data_root": str(root), "archive_root": str(archive_root),
                   "output_root": str(root), "event_schema": {"relations": ["EVENT_WRITE"]}})
            archive = Archive(archive_root)
            self.assertEqual(archive.raw_event(0)["event_id"], "event-1")
            archive.close()


if __name__ == "__main__":
    unittest.main()
