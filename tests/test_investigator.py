import os
import unittest
from unittest.mock import Mock, patch

from lode.investigator import _call, _interleave_unique, _messages, run_case, validate


class InvestigatorValidationTests(unittest.TestCase):
    def test_supported_direct_claim(self):
        seed = {"id": 0, "src": 1, "dst": 2, "rel": "EVENT_EXECUTE", "ts": 1}
        event = {"id": 1, "src": 2, "dst": 3, "rel": "EVENT_WRITE", "ts": 2}
        answer = {"select": [0, 1], "direct": [{"node": 3, "support": [1]}]}
        selected, direct = validate(answer, seed, {}, [event])
        self.assertEqual(set(selected), {0, 1})
        self.assertEqual(direct[0]["node"], 3)

    def test_rejects_unseen_event(self):
        seed = {"id": 0, "src": 1, "dst": 2, "rel": "EVENT_EXECUTE", "ts": 1}
        with self.assertRaises(ValueError):
            validate({"select": [99], "direct": []}, seed, {}, [])

    def test_message_budget_is_measured_in_tokens(self):
        archive = Mock()
        archive.attributes.return_value = "/" + "long-path/" * 20
        seed = {"id": 0, "src": 1, "dst": 2, "rel": "EVENT_EXECUTE", "ts": 1}
        packet = [
            {"id": i, "src": 1, "dst": i + 2, "rel": "EVENT_WRITE", "ts": i}
            for i in range(1, 30)
        ]
        messages, trimmed, tokens = _messages(
            archive, seed, {}, packet, [], budget=512, tokenizer="o200k_base"
        )
        self.assertLessEqual(tokens, 512)
        self.assertLess(len(trimmed), len(packet))
        self.assertEqual(messages[0]["role"], "system")

    def test_initial_page_interleaving_keeps_unequal_tail(self):
        first = [{"id": 1}, {"id": 2}]
        second = [{"id": 3}]
        self.assertEqual(
            [event["id"] for event in _interleave_unique(first, second, 4)],
            [1, 3, 2],
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("lode.investigator.requests.post")
    def test_call_sends_fixed_temperature(self, post):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": '{"select":[],"direct":[],"done":true}'}}],
            "usage": {"prompt_tokens": 12},
            "model": "gpt-5.5-snapshot",
        }
        post.return_value = response
        config = {"investigator": {
            "model": "gpt-5.5", "max_output_tokens": 64,
            "temperature": 0.0, "timeout_seconds": 90,
        }}
        answer, metadata = _call([{"role": "user", "content": "{}"}], config)
        self.assertTrue(answer["done"])
        self.assertEqual(post.call_args.kwargs["json"]["temperature"], 0.0)
        self.assertEqual(metadata["model"], "gpt-5.5-snapshot")

    @patch("lode.investigator._call")
    def test_run_case_enforces_retrieval_page_budget(self, call):
        call.return_value = (
            {"select": [0], "direct": [], "done": False,
             "query": {"node": 1, "anchor": 0, "side": "past", "order": "near"}},
            {"usage": {}, "model": "gpt-5.5"},
        )

        class ArchiveStub:
            def __init__(self):
                self.query_id = 10

            def event(self, position):
                return {"id": int(position), "src": 1, "dst": 2,
                        "rel": "EVENT_EXECUTE", "ts": 10}

            def attributes(self, node):
                return f"node-{node}"

            def incident(self, node, anchor, side, order, limit=24, cursor=None):
                meta = {"node": node, "anchor": anchor, "side": side, "order": order,
                        "rows_examined": 1, "capped_scan": False,
                        "returned": 0, "cursor": self.query_id, "latency_s": 0.0}
                if limit == 4096:
                    return [], meta
                event = {"id": self.query_id, "src": 1, "dst": self.query_id,
                         "rel": "EVENT_READ", "ts": self.query_id}
                self.query_id += 1
                meta["returned"] = 1
                return [event], meta

        config = {
            "retrieval": {"packet_events": 8, "pages_per_seed": 3},
            "investigator": {"max_prompt_tokens": 4096, "tokenizer": "o200k_base"},
        }
        result = run_case(ArchiveStub(), {"anchor": 0, "owner": 1}, config)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(len(result["retrieval_pages"]), 3)
        self.assertEqual(result["stop_reason"], "page_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
