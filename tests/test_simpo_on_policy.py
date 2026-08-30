import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "local"))

from build_simpo_pairs import build_pairs
from merge_simpo_on_policy_samples import merge_samples
from score_nbest_oracle import score_file


class SimPOOnPolicyTest(unittest.TestCase):
    def test_merge_drops_only_all_identical_prompts(self):
        seeds = [13, 21, 42, 79, 100]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            inputs = []
            for index, seed in enumerate(seeds):
                path = tmp / f"seed_{seed}.jsonl"
                rows = [
                    {
                        "text_id": "varied",
                        "query": "q1",
                        "audio": "a1.wav",
                        "prompt": "p",
                        "text": "t1",
                        "semantics": [],
                        "nbest": [f"response-{index % 3}"],
                    },
                    {
                        "text_id": "identical",
                        "query": "q2",
                        "audio": "a2.wav",
                        "prompt": "p",
                        "text": "t2",
                        "semantics": [],
                        "nbest": ["same-response"],
                    },
                ]
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                inputs.append(str(path))

            output = tmp / "merged.jsonl"
            stats = merge_samples(inputs, seeds, str(output))
            merged = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(stats["total_prompts"], 2)
            self.assertEqual(stats["retained_prompts"], 1)
            self.assertEqual(stats["dropped_all_identical"], 1)
            self.assertEqual(merged[0]["text_id"], "varied")
            self.assertEqual(merged[0]["nbest_seeds"], seeds)
            self.assertEqual(len(merged[0]["nbest"]), 5)

    def test_highest_lowest_ignores_ground_truth_and_margin_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            scored_path = tmp / "scored.jsonl"
            output_path = tmp / "pairs.jsonl"
            row = {
                "text_id": "sample",
                "query": "q",
                "audio": "a.wav",
                "prompt": "p",
                "semantics": [],
                "ground_truth_candidate": {
                    "raw": "ground-truth",
                    "preference_score": 999.0,
                },
                "scored_nbest": [
                    {
                        "rank": 0,
                        "raw": "middle",
                        "score": {"valid_json": 1},
                        "preference_score": 0.5,
                    },
                    {
                        "rank": 1,
                        "raw": "winner",
                        "score": {"valid_json": 1},
                        "preference_score": 0.9,
                    },
                    {
                        "rank": 2,
                        "raw": "rejected-invalid-json",
                        "score": {"valid_json": 0},
                        "preference_score": 0.1,
                    },
                ],
            }
            scored_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            stats = build_pairs(
                str(scored_path),
                str(output_path),
                min_score_margin=999.0,
                max_pairs_per_sample=9,
                pair_mode="sampled_highest_lowest",
            )
            pair = json.loads(output_path.read_text().strip())

            self.assertEqual(stats["pairs"], 1)
            self.assertEqual(pair["chosen"], "winner")
            self.assertEqual(pair["rejected"], "rejected-invalid-json")
            self.assertEqual(pair["chosen_source"], "sampled_highest")

    def test_merged_nbest_schema_supports_oracle_ema_coverage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "merged.jsonl"
            output_path = tmp / "scored.jsonl"
            oracle_semantics = [{"domain": "home", "intent": "lights_on", "slots": {}}]
            wrong_semantics = [{"domain": "home", "intent": "lights_off", "slots": {}}]

            def response(semantics):
                return json.dumps({"asr_text": "turn on lights", "semantics": semantics})

            rows = [
                {
                    "text_id": "oracle-hit",
                    "query": "turn on lights",
                    "semantics": oracle_semantics,
                    "nbest": [response(wrong_semantics), response(oracle_semantics)],
                    "nbest_seeds": [13, 21],
                },
                {
                    "text_id": "oracle-miss",
                    "query": "turn on lights",
                    "semantics": oracle_semantics,
                    "nbest": [response(wrong_semantics), response(wrong_semantics)],
                    "nbest_seeds": [13, 21],
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            stats = score_file(str(input_path), str(output_path))

            self.assertEqual(stats["samples"], 2)
            self.assertEqual(stats["oracle_hit_samples"], 1)
            self.assertEqual(stats["oracle_ema_coverage"], 0.5)

    def test_oracle_highest_lowest_requires_sampled_oracle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            scored_path = tmp / "scored.jsonl"
            output_path = tmp / "pairs.jsonl"
            rows = [
                {
                    "text_id": "oracle-hit",
                    "scored_nbest": [
                        {
                            "rank": 0,
                            "raw": "nonoracle-middle",
                            "score": {"oracle_ema": 0},
                            "preference_score": 10.0,
                        },
                        {
                            "rank": 1,
                            "raw": "sampled-oracle",
                            "score": {"oracle_ema": 1},
                            "preference_score": 1000.0,
                        },
                        {
                            "rank": 2,
                            "raw": "nonoracle-worst",
                            "score": {"oracle_ema": 0},
                            "preference_score": 1.0,
                        },
                    ],
                },
                {
                    "text_id": "oracle-miss",
                    "ground_truth_candidate": {
                        "raw": "must-not-be-used",
                        "score": {"oracle_ema": 1},
                        "preference_score": 1000.0,
                    },
                    "scored_nbest": [
                        {
                            "rank": 0,
                            "raw": "wrong-a",
                            "score": {"oracle_ema": 0},
                            "preference_score": 2.0,
                        },
                        {
                            "rank": 1,
                            "raw": "wrong-b",
                            "score": {"oracle_ema": 0},
                            "preference_score": 1.0,
                        },
                    ],
                },
            ]
            scored_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            stats = build_pairs(
                str(scored_path),
                str(output_path),
                min_score_margin=999.0,
                max_pairs_per_sample=9,
                pair_mode="oracle_sampled_highest_lowest",
            )
            pairs = [json.loads(line) for line in output_path.read_text().splitlines()]

            self.assertEqual(stats["pairs"], 1)
            self.assertEqual(stats["dropped_no_sampled_oracle"], 1)
            self.assertEqual(pairs[0]["chosen"], "sampled-oracle")
            self.assertEqual(pairs[0]["rejected"], "nonoracle-worst")
            self.assertEqual(pairs[0]["chosen_source"], "sampled_oracle")


if __name__ == "__main__":
    unittest.main()
