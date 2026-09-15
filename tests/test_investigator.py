import unittest

from lode.investigator import validate


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


if __name__ == "__main__":
    unittest.main()
