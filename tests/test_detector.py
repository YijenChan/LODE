import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lode.detector import EventGRU, next_event_loss, owner_and_signature, prepare, process_windows


class GRUDetectorTests(unittest.TestCase):
    def test_owner_and_typed_signature(self):
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 3])
        types = np.array([0, 1, 0, 0])
        relation = np.array([1, 2, 0])
        owner, token = owner_and_signature(src, dst, types, relation, clone_relation=0)
        np.testing.assert_array_equal(owner, [0, 2, 2])
        np.testing.assert_array_equal(token, [8, 15, 0])

    def test_next_event_interface(self):
        model = EventGRU(vocabulary=12, embedding_dim=4, hidden_dim=5)
        target = torch.tensor([[1, 2, 3], [4, 5, 0]])
        hidden = torch.zeros(1, 2, 5)
        loss, next_hidden, count = next_event_loss(model, target, np.array([3, 2]), hidden)
        self.assertEqual(count, 5)
        self.assertEqual(tuple(next_hidden.shape), (1, 2, 5))
        self.assertTrue(torch.isfinite(loss))

    def test_process_window_aggregation(self):
        positions = np.array([0, 1, 2, 3], dtype=np.int32)
        offsets = np.array([0, 2, 4], dtype=np.int64)
        timestamps = np.array([0, 11, 1, 21], dtype=np.int64)
        scores = np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float32)
        values, records = process_windows(positions, offsets, timestamps, scores, 10)
        np.testing.assert_array_equal(values, [1.0, 3.0, 2.0, 4.0])
        self.assertEqual(records[0]["anchor"], 1)
        self.assertEqual(records[1]["anchor"], 3)

    def test_prepare_reads_canonical_parquet_without_baseline_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "canonical" / "train" / "sorted"
            split.mkdir(parents=True)
            pq.write_table(pa.table({
                "node_uuid": ["p", "f"], "node_type": ["process", "file"]
            }), split / "nodes.parquet")
            pq.write_table(pa.table({
                "src_uuid": ["p"], "relation": ["EVENT_WRITE"],
                "dst_uuid": ["f"], "timestamp_ns": [1]
            }), split / "events.parquet")
            config = {
                "run_id": "test", "data_root": str(root),
                "output_root": str(root / "runs"),
                "event_schema": {
                    "entity_types": ["process", "file", "netflow"],
                    "relations": ["EVENT_CLONE", "EVENT_WRITE"],
                },
            }
            prepare(config, "train")
            signature = np.load(root / "runs" / "test" / "middleware" / "train" / "signature.npy")
            self.assertEqual(signature.tolist(), [8])


if __name__ == "__main__":
    unittest.main()
