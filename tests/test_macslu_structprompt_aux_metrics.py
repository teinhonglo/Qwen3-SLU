import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "local"))

from metrics_macslu_structprompt_aux import evaluate_cdi, evaluate_pii


class StructPromptAuxMetricsTest(unittest.TestCase):
    def test_pii_relation_metrics(self):
        gt_rows = [
            {
                "text_id": "a__pii",
                "text": (
                    'language None<asr_text>['
                    '{"domain":"music","intent":"play","slots":["song"]},'
                    '{"domain":"map","intent":"nav","slots":[]}]'
                ),
                "pii_domain_intents": [
                    {"domain": "music", "intent": "play"},
                    {"domain": "map", "intent": "nav"},
                ],
                "pii_slots": ["song"],
            },
            {
                "text_id": "b__pii",
                "text": (
                    'language None<asr_text>['
                    '{"domain":"music","intent":"play","slots":["artist"]}]'
                ),
                "pii_domain_intents": [
                    {"domain": "music", "intent": "play"},
                ],
                "pii_slots": ["artist"],
            },
        ]
        pred_rows = [
            {
                "text_id": "a__pii",
                "pred_raw": (
                    'language None<asr_text>['
                    '{"domain":"map","intent":"nav","slots":[]},'
                    '{"domain":"music","intent":"play","slots":["song"]}]'
                ),
            },
            {
                "text_id": "b__pii",
                "pred_raw": (
                    'language None<asr_text>['
                    '{"domain":"music","intent":"play","slots":[]}]'
                ),
            },
        ]

        metrics, _ = evaluate_pii(pred_rows, gt_rows)
        self.assertEqual(metrics["num_examples"], 2)
        self.assertAlmostEqual(metrics["valid_json_rate"], 1.0)
        self.assertAlmostEqual(metrics["candidate_valid_rate"], 1.0)
        self.assertAlmostEqual(metrics["relation_exact_match"], 0.5)
        self.assertAlmostEqual(metrics["domain_intent_micro"]["f1"], 1.0)
        self.assertAlmostEqual(metrics["intent_slot_pair_micro"]["precision"], 1.0)
        self.assertAlmostEqual(metrics["intent_slot_pair_micro"]["recall"], 0.5)

    def test_cdi_invalid_output_counts_as_error(self):
        gt_rows = [
            {"text_id": "a", "cdi_label": True, "current_intent_count": 1, "reference_intent_count": 1},
            {"text_id": "b", "cdi_label": False, "current_intent_count": 1, "reference_intent_count": 2},
            {"text_id": "c", "cdi_label": True, "current_intent_count": 2, "reference_intent_count": 2},
            {"text_id": "d", "cdi_label": False, "current_intent_count": 2, "reference_intent_count": 1},
        ]
        pred_rows = [
            {"text_id": "a", "pred_raw": "language None<asr_text>true"},
            {"text_id": "b", "pred_raw": "false"},
            {"text_id": "c", "pred_raw": "maybe"},
            {"text_id": "d", "pred_raw": "language None<asr_text>false"},
        ]

        metrics, _ = evaluate_cdi(pred_rows, gt_rows)
        self.assertAlmostEqual(metrics["valid_output_rate"], 0.75)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["accuracy_on_valid_outputs"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)
        self.assertEqual(metrics["invalid_outputs"], 1)
        self.assertEqual(metrics["true_class"]["tp"], 1)
        self.assertEqual(metrics["true_class"]["fn"], 1)
        self.assertEqual(metrics["false_class"]["tp"], 2)


if __name__ == "__main__":
    unittest.main()
