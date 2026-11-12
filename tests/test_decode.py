from bolo.adapt import Memory
from bolo.asr.decode import TrieBoost


class Tok:
    """Tiny fake tokenizer: one token per character."""

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]


def test_trie_boost_follows_phrases():
    import torch

    tb = TrieBoost(Tok(), ["ab"], bonus=2.0, start_bonus=0.5)
    scores = torch.zeros(1, 200)
    out = tb(torch.tensor([[ord(" ")]]), scores.clone())  # after " " the next "a" continues " ab"
    assert out[0, ord("a")] == 2.0 and out[0, ord(" ")] == 0.5 and out[0, ord("b")] == 0.0
    out = tb(torch.tensor([[ord(" "), ord("a")]]), scores.clone())
    assert out[0, ord("b")] == 2.0


def test_memory_learns_alias_and_weights():
    m = Memory()
    cands = [
        {"key": ("open_app", (("app", "doctor"),)), "pred": {"intent": "open_app", "slots": {"app": "doctor"}},
         "heard": {"app": "doctor"}, "f": {"asr": -0.1, "nlu": 0.0, "ground": 0.5, "none": 0.0, "prior": 0.0}, "s": 0.4},
        {"key": ("open_app", (("app", "docker"),)), "pred": {"intent": "open_app", "slots": {"app": "docker"}},
         "heard": {"app": "docker"}, "f": {"asr": -0.3, "nlu": 0.0, "ground": 1.0, "none": 0.0, "prior": 0.0}, "s": 0.2},
    ]
    w0 = m.w["ground"]
    m.feedback(cands, {"intent": "open_app", "slots": {"app": "docker"}}, cands[0])
    assert m.aliases["app:doctor"] == "docker"
    assert m.w["ground"] > w0 and m.counts["docker"] == 1
    assert m.apply_aliases({"intent": "open_app", "slots": {"app": "Doctor"}})["slots"]["app"] == "docker"
