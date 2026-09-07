import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "local"))

from prepare_macslu_structprompt_jsonl import build_cdi_rows, convert_split


def make_row(index: int, intent_count: int) -> dict:
    semantics = [
        {
            "domain": "domain",
            "intent": f"intent_{intent_index}",
            "slots": {"slot": f"value_{index}_{intent_index}"},
            "implicit_slots": {},
        }
        for intent_index in range(intent_count)
    ]
    return {
        "text_id": f"id_{index}",
        "query": f"query {index}",
        "audio": f"audio_{index}.wav",
        "prompt": "prompt",
        "text": "target",
        "semantics": semantics,
    }


class PrepareMacSLUStructPromptTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            make_row(0, 1),
            make_row(1, 1),
            make_row(2, 2),
            make_row(3, 2),
        ]

    def test_samples_one_positive_and_one_negative_per_anchor(self):
        grouped = build_cdi_rows(
            self.rows, random.Random(67), pairs_per_class=1
        )

        self.assertEqual(len(grouped), len(self.rows))
        for anchor, pairs in zip(self.rows, grouped):
            self.assertEqual(len(pairs), 2)
            self.assertEqual({pair["cdi_label"] for pair in pairs}, {True, False})
            self.assertEqual(
                len({pair["reference_text_id"] for pair in pairs}), 2
            )
            self.assertNotIn(
                anchor["text_id"],
                {pair["reference_text_id"] for pair in pairs},
            )
            for pair in pairs:
                same_count = (
                    pair["current_intent_count"]
                    == pair["reference_intent_count"]
                )
                self.assertEqual(pair["cdi_label"], same_count)

    def test_repeats_slu_and_pii_to_match_cdi_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "train.jsonl"
            output = root / "expanded.jsonl"
            source.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows
                ),
                encoding="utf-8",
            )

            counts = convert_split(
                source,
                output,
                expand=True,
                seed=66,
                cdi_pairs_per_class=1,
            )
            expanded = [json.loads(line) for line in output.read_text().splitlines()]

            expected_per_task = len(self.rows) * 2
            self.assertEqual(counts["slu"], expected_per_task)
            self.assertEqual(counts["pii"], expected_per_task)
            self.assertEqual(counts["cdi"], expected_per_task)
            self.assertEqual(counts["cdi_true"], len(self.rows))
            self.assertEqual(counts["cdi_false"], len(self.rows))
            self.assertEqual(counts["total"], len(self.rows) * 6)
            self.assertEqual(
                {row["text_id"] for row in expanded if row["task"] == "slu"},
                {row["text_id"] for row in self.rows},
            )

    def test_count_above_three_is_negative_reference_but_not_anchor(self):
        rows = [make_row(0, 3), make_row(1, 3), make_row(2, 4)]
        grouped = build_cdi_rows(
            rows,
            random.Random(67),
            pairs_per_class=1,
            max_anchor_intent_count=3,
        )

        self.assertEqual([len(group) for group in grouped], [2, 2, 0])
        for group in grouped[:2]:
            negative = next(pair for pair in group if not pair["cdi_label"])
            self.assertEqual(negative["reference_intent_count"], 4)

    def test_balancing_preserves_rows_above_anchor_limit_in_slu_and_pii(self):
        rows = [make_row(0, 3), make_row(1, 3), make_row(2, 4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "train.jsonl"
            output = root / "expanded.jsonl"
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            counts = convert_split(
                source,
                output,
                expand=True,
                seed=66,
                cdi_pairs_per_class=1,
                cdi_max_anchor_intent_count=3,
            )
            expanded = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(counts["cdi_anchors"], 2)
            self.assertEqual(counts["slu"], 4)
            self.assertEqual(counts["pii"], 4)
            self.assertEqual(counts["cdi"], 4)
            self.assertIn(
                "id_2",
                {row["text_id"] for row in expanded if row["task"] == "slu"},
            )
            self.assertIn(
                "id_2__pii",
                {row["text_id"] for row in expanded if row["task"] == "pii"},
            )

    def test_rejects_k_larger_than_available_positive_pool(self):
        with self.assertRaisesRegex(ValueError, "Not enough positive CDI references"):
            build_cdi_rows(self.rows, random.Random(67), pairs_per_class=2)


if __name__ == "__main__":
    unittest.main()
