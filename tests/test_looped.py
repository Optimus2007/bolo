from bolo.pipeline import looped


def test_looped():
    assert looped("cpu cpu cpu cpu cpu cpu")
    assert looped("volume kam karo volume kam karo volume kam karo")
    for ok in ("volume kam karo", "bhai bhai terminal kholo", "rahul ko call karo na yaar abhi please",
               "chalo 30 ghante mein remind karna ki notes padhne hain"):
        assert not looped(ok)


def test_guards():
    import numpy as np

    from bolo.asr.guards import energy_speech_seconds as speech_seconds, max_tokens, rate_ok

    sr = 16000
    silence = np.zeros(sr * 2, "float32") + np.random.default_rng(0).normal(0, 1e-4, sr * 2).astype("float32")
    tone = silence.copy()
    tone[sr // 2: sr // 2 + sr] += 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype("float32")
    assert speech_seconds(silence) < 0.25
    assert 0.8 < speech_seconds(tone) < 1.2
    assert max_tokens(1.5) >= 20
    assert rate_ok("volume kam karo", 1.0) and not rate_ok("cpu " * 20, 1.0) and not rate_ok("ok", 3.0)
