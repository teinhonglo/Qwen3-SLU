import unittest

from slu_decoding.state_parser import (
    STATE_AFTER_DOMAIN,
    STATE_AFTER_IMPLICIT_SLOTS,
    STATE_AFTER_INTENT,
    STATE_AFTER_SLOT_KEY,
    STATE_AFTER_SLOTS,
    STATE_COMPLETE,
    STATE_DOMAIN,
    STATE_IMPLICIT_SLOTS_KEY,
    STATE_IMPLICIT_SLOTS_VALUE,
    STATE_INTENT,
    STATE_SEMANTICS_START,
    STATE_SLOTS_KEY,
    STATE_SLOTS_NEXT,
    STATE_SLOTS_VALUE,
    parse_state,
)
from slu_decoding.token_trie import TokenIDTrie


class StateParserTest(unittest.TestCase):
    def test_tracks_active_escaped_labels(self):
        prefix = (
            '<asr_text>{"asr_text":"带我去广州塔","semantics":"'
            '[{\\"domain\\":\\"地图\\",\\"intent\\":\\"导航\\",'
            '\\"slots\\":{\\"终点'
        )
        state = parse_state(prefix)
        self.assertEqual(state.state_name, STATE_SLOTS_KEY)
        self.assertEqual(state.current_domain, "地图")
        self.assertEqual(state.current_intent, "导航")
        self.assertEqual(state.active_label_prefix, "终点")
        self.assertEqual(state.active_label_quote, '\\"')
        self.assertTrue(state.active_label)

    def test_distinguishes_slot_key_and_value(self):
        key_prefix = (
            '<asr_text>{"semantics":[{"domain":"地图","intent":"导航",'
            '"slots":{"终'
        )
        value_prefix = key_prefix[:-1] + '终点目标":"广'

        key_state = parse_state(key_prefix)
        value_state = parse_state(value_prefix)

        self.assertEqual(key_state.state_name, STATE_SLOTS_KEY)
        self.assertEqual(key_state.active_label_prefix, "终")
        self.assertEqual(value_state.state_name, STATE_SLOTS_VALUE)
        self.assertEqual(value_state.current_slot_key, "终点目标")
        self.assertFalse(value_state.active_label)

    def test_uses_latest_frame_in_multi_intent_output(self):
        prefix = (
            '<asr_text>{"asr_text":"x","semantics":"'
            '[{\\"domain\\":\\"地图\\",\\"intent\\":\\"导航\\",'
            '\\"slots\\":{},\\"implicit_slots\\":{}},'
            '{\\"domain\\":\\"音乐\\",\\"intent\\":\\"播放音乐\\",'
            '\\"slots\\":{\\"歌'
        )
        state = parse_state(prefix)
        self.assertEqual(state.state_name, STATE_SLOTS_KEY)
        self.assertEqual(state.current_domain, "音乐")
        self.assertEqual(state.current_intent, "播放音乐")
        self.assertEqual(state.active_label_prefix, "歌")

    def test_tracks_implicit_slot_key_separately(self):
        prefix = (
            '<asr_text>{"semantics":[{"domain":"车辆控制",'
            '"intent":"空调控制","slots":{},"implicit_slots":{"温'
        )
        state = parse_state(prefix)
        self.assertEqual(state.state_name, STATE_IMPLICIT_SLOTS_KEY)
        self.assertEqual(state.active_label_prefix, "温")

    def test_tracks_domain_and_intent_prefixes(self):
        domain_state = parse_state(
            '<asr_text>{"semantics":[{"domain":"地'
        )
        intent_state = parse_state(
            '<asr_text>{"semantics":[{"domain":"地图","intent":"导'
        )
        self.assertEqual(domain_state.state_name, STATE_DOMAIN)
        self.assertEqual(domain_state.active_label_prefix, "地")
        self.assertEqual(intent_state.state_name, STATE_INTENT)
        self.assertEqual(intent_state.current_domain, "地图")
        self.assertEqual(intent_state.active_label_prefix, "导")

    def test_tracks_escaped_json_structure_after_domain(self):
        base = (
            '<asr_text>{"asr_text": "关了车窗", "semantics": "'
            '[{\\"domain\\": \\"车载控制\\"'
        )
        transition = ', \\"intent\\": \\"'

        for length in range(len(transition)):
            state = parse_state(base + transition[:length])
            self.assertEqual(state.state_name, STATE_AFTER_DOMAIN)
            self.assertTrue(state.active_structure)
            self.assertEqual(state.active_structure_prefix, transition[:length])
            self.assertEqual(state.structure_candidates, (transition,))

        state = parse_state(base + transition)
        self.assertEqual(state.state_name, STATE_INTENT)
        self.assertTrue(state.active_label)

    def test_tracks_full_escaped_json_grammar(self):
        prefixes = {
            '<asr_text>{"asr_text": "x", "semantics": "': STATE_SEMANTICS_START,
            '<asr_text>{"asr_text": "x", "semantics": "'
            '[{\\"domain\\": \\"车载控制\\"': STATE_AFTER_DOMAIN,
            '<asr_text>{"asr_text": "x", "semantics": "'
            '[{\\"domain\\": \\"车载控制\\", '
            '\\"intent\\": \\"车身控制\\"': STATE_AFTER_INTENT,
            '<asr_text>{"asr_text": "x", "semantics": "'
            '[{\\"domain\\": \\"车载控制\\", '
            '\\"intent\\": \\"车身控制\\", \\"slots\\": {'
            '\\"操作\\"': STATE_AFTER_SLOT_KEY,
            '<asr_text>{"asr_text": "x", "semantics": "'
            '[{\\"domain\\": \\"车载控制\\", '
            '\\"intent\\": \\"车身控制\\", '
            '\\"slots\\": {\\"操作\\": \\"关了\\"': STATE_SLOTS_NEXT,
            '<asr_text>{"asr_text": "x", "semantics": "'
            '[{\\"domain\\": \\"车载控制\\", '
            '\\"intent\\": \\"车身控制\\", '
            '\\"slots\\": {\\"操作\\": \\"关了\\"}': STATE_AFTER_SLOTS,
            '<asr_text>{"asr_text": "x", "semantics": "'
            '[{\\"domain\\": \\"车载控制\\", '
            '\\"intent\\": \\"车身控制\\", '
            '\\"slots\\": {}, \\"implicit_slots\\": {}': STATE_AFTER_IMPLICIT_SLOTS,
        }
        for prefix, expected_state in prefixes.items():
            state = parse_state(prefix)
            self.assertEqual(state.state_name, expected_state)
            self.assertTrue(state.active_structure)

    def test_recognizes_complete_multi_frame_target(self):
        target = (
            'language None<asr_text>{"asr_text": "关了车窗打开内循环", '
            '"semantics": "[{\\"domain\\": \\"车载控制\\", '
            '\\"intent\\": \\"车身控制\\", '
            '\\"slots\\": {\\"操作\\": \\"关了\\", '
            '\\"对象\\": \\"车窗\\"}, \\"implicit_slots\\": {}}, '
            '{\\"domain\\": \\"车载控制\\", '
            '\\"intent\\": \\"提供信息\\", '
            '\\"slots\\": {\\"操作\\": \\"打开\\", '
            '\\"模式\\": \\"内循环\\"}, '
            '\\"implicit_slots\\": {\\"调节内容\\": \\"模式\\"}}]"}'
        )
        self.assertEqual(parse_state(target).state_name, STATE_COMPLETE)

        semantics_start = target.index('"semantics": "') + len('"semantics": "')
        for end in range(semantics_start, len(target) + 1):
            state = parse_state(target[:end])
            self.assertTrue(
                state.active_structure
                or state.active_label
                or state.state_name
                in (STATE_SLOTS_VALUE, STATE_IMPLICIT_SLOTS_VALUE, STATE_COMPLETE),
                msg=f"unconstrained semantic prefix at offset {end}: {target[:end]!r}",
            )


class TokenIDTrieTest(unittest.TestCase):
    def test_returns_children_for_shared_token_prefix(self):
        trie = TokenIDTrie()
        trie.insert([10, 20, 30])
        trie.insert([10, 20, 40])
        trie.insert([10, 50])

        self.assertEqual(trie.next_token_ids([]), {10})
        self.assertEqual(trie.next_token_ids([10]), {20, 50})
        self.assertEqual(trie.next_token_ids([10, 20]), {30, 40})

    def test_distinguishes_terminal_and_missing_prefix(self):
        trie = TokenIDTrie()
        trie.insert([1, 2])

        self.assertEqual(trie.next_token_ids([1, 2]), set())
        self.assertIsNone(trie.next_token_ids([9]))
        self.assertEqual(trie.max_depth, 2)


if __name__ == "__main__":
    unittest.main()
