import json
import os
import sys
import tempfile
import types
import unittest

from local.build_macslu_schema import build_schema

# The repository's production inference environment provides torch and
# transformers.  This unit test exercises the tokenizer/schema-only replay path,
# so tiny import stubs keep that path testable in CPU-light development images.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")
try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.LogitsProcessor = type("LogitsProcessor", (), {})
    sys.modules["transformers"] = transformers_stub

from slu_decoding.logits_processors import StateAwareDExpertsLogitsProcessor
from slu_decoding.schema import SLUSchema


def make_row(text_id, query, frames):
    payload = json.dumps(
        {
            "asr_text": query,
            "semantics": json.dumps(frames, ensure_ascii=False),
        },
        ensure_ascii=False,
    )
    return {
        "text_id": text_id,
        "query": query,
        "semantics": frames,
        "text": f"language None<asr_text>{payload}",
    }


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return "".join(chr(int(token_id) - 1) for token_id in token_ids if token_id)


class DataDrivenConstraintTest(unittest.TestCase):
    def setUp(self):
        self.frames = [
            {
                "domain": "车载控制",
                "intent": "车身控制",
                "slots": {"操作": "关了", "对象": "车窗"},
                "implicit_slots": {},
            },
            {
                "domain": "车载控制",
                "intent": "提供信息",
                "slots": {"操作": "打开", "模式": "内循环"},
                "implicit_slots": {"调节内容": "模式"},
            },
        ]
        self.row = make_row("id_1", "关了车窗打开内循环", self.frames)

    def _write_schema(self, rows):
        directory = tempfile.TemporaryDirectory()
        jsonl_path = os.path.join(directory.name, "train.jsonl")
        schema_path = os.path.join(directory.name, "schema.json")
        with open(jsonl_path, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        schema_data = build_schema([jsonl_path])
        with open(schema_path, "w", encoding="utf-8") as stream:
            json.dump(schema_data, stream, ensure_ascii=False)
        return directory, schema_path, schema_data

    def test_schema_structure_literals_come_from_real_target_text(self):
        directory, _, schema = self._write_schema([self.row])
        self.addCleanup(directory.cleanup)
        forms = schema["constraint_surface_forms"]

        self.assertEqual(forms["domain_followups"], ['\\", \\"intent\\": \\"'])
        self.assertEqual(forms["slots_next"], [', \\"', '}'])
        self.assertEqual(
            set(forms["after_implicit_slots"]),
            {'}]"}', '}, {\\"domain\\": \\"'},
        )

    def test_gold_token_replay_accepts_every_constrained_token(self):
        directory, schema_path, _ = self._write_schema([self.row])
        self.addCleanup(directory.cleanup)
        processor = StateAwareDExpertsLogitsProcessor(
            CharacterTokenizer(),
            schema=SLUSchema(schema_path),
            schema_constraint_mode="hard",
            enable_grounding=False,
        )

        replay = processor.validate_gold_targets([self.row["text"]])

        self.assertTrue(replay["ok"], replay)
        self.assertGreater(replay["constrained_tokens"], 0)

    def test_schema_keeps_each_serialization_style_observed_in_data(self):
        compact_semantics = json.dumps(
            self.frames, ensure_ascii=False, separators=(",", ":")
        )
        compact_payload = json.dumps(
            {
                "asr_text": self.row["query"],
                "semantics": compact_semantics,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        compact_row = dict(self.row)
        compact_row["text_id"] = "id_compact"
        compact_row["text"] = f"language None<asr_text>{compact_payload}"
        directory, _, schema = self._write_schema([self.row, compact_row])
        self.addCleanup(directory.cleanup)

        self.assertEqual(
            set(schema["constraint_surface_forms"]["domain_followups"]),
            {'\\", \\"intent\\": \\"', '\\",\\"intent\\":\\"'},
        )

    def test_schema_build_fails_if_text_and_semantics_disagree(self):
        bad_row = dict(self.row)
        bad_row["semantics"] = []
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        jsonl_path = os.path.join(directory.name, "train.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(bad_row, ensure_ascii=False) + "\n")

        with self.assertRaisesRegex(ValueError, "text target semantics disagree"):
            build_schema([jsonl_path])


if __name__ == "__main__":
    unittest.main()
