class TokenIDTrie:
    """Prefix tree whose edges are tokenizer vocabulary IDs."""

    _END = object()

    def __init__(self):
        self.root = {}
        self.max_depth = 0

    def insert(self, token_ids):
        token_ids = [int(token_id) for token_id in token_ids]
        if not token_ids:
            return

        node = self.root
        for token_id in token_ids:
            node = node.setdefault(token_id, {})
        node[self._END] = True
        self.max_depth = max(self.max_depth, len(token_ids))

    def next_token_ids(self, prefix_ids):
        """Return children after ``prefix_ids``, or ``None`` for a missing path."""
        node = self.root
        for token_id in prefix_ids:
            token_id = int(token_id)
            if token_id not in node:
                return None
            node = node[token_id]
        return {token_id for token_id in node if token_id is not self._END}


def remaining_text_candidates(candidates, decoded_prefix):
    """Return candidate suffixes after an already-decoded text prefix.

    The decoded prefix may begin in the middle of the most recently generated
    tokenizer token.  Matching in text space avoids assuming that a grammar
    state transition is also a tokenizer-token boundary.
    """
    prefix = decoded_prefix or ""
    remaining = []
    for candidate in candidates:
        if candidate.startswith(prefix):
            suffix = candidate[len(prefix) :]
            if suffix:
                remaining.append(suffix)
    return remaining
