import numpy as np

from bolo.audio import augment as A


def tone(sec=1.0, f=440.0):
    t = np.arange(int(A.SR * sec)) / A.SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_wav_roundtrip(tmp_path):
    x = tone()
    A.write_wav(tmp_path / "a.wav", x)
    y, sr = A.read_wav(str(tmp_path / "a.wav"))
    assert sr == 16000 and len(y) == len(x) and np.max(np.abs(x - y)) < 1e-3


def test_resample_preserves_length_and_pitch():
    x = tone(1.0, 440)
    y = A.resample(x, 16000, 24000)
    assert len(y) == 24000
    peak = np.argmax(np.abs(np.fft.rfft(y))) * 24000 / len(y)
    assert abs(peak - 440) < 2
    assert len(A.resample(A.resample(x, 16000, 44100), 44100, 16000)) == 16000


def test_speed_changes_length():
    x = tone(1.0)
    assert len(A.speed(x, 1.1)) == round(16000 / 1.1)
    assert len(A.speed(x, 0.9)) == round(16000 / 0.9)


def test_noise_hits_target_snr():
    rng = np.random.default_rng(0)
    x = tone()
    n = A.coloured_noise(rng, len(x), "pink")
    y = A.add_noise(x, n, 10.0)
    snr = 20 * np.log10(A.rms(x) / A.rms(y - x))
    assert abs(snr - 10.0) < 0.1


def test_bandlimit_removes_low_and_high():
    lo, mid, hi = tone(f=100), tone(f=1000), tone(f=6000)
    assert A.rms(A.bandlimit(lo)) < 0.05 * A.rms(lo)
    assert A.rms(A.bandlimit(hi)) < 0.05 * A.rms(hi)
    assert A.rms(A.bandlimit(mid)) > 0.9 * A.rms(mid)


def test_reverb_and_gain_safe():
    rng = np.random.default_rng(1)
    x = tone()
    y = A.reverb(x, A.room_ir(rng, 0.5))
    assert len(y) == len(x) and np.isfinite(y).all()
    assert np.max(np.abs(A.gain(x * 3, 6))) <= 0.99 + 1e-6


def test_augment_deterministic_and_bounded():
    x = tone()
    a, pa = A.augment(x, np.random.default_rng(5))
    b, pb = A.augment(x, np.random.default_rng(5))
    assert pa == pb and np.array_equal(a, b)
    for s in range(30):
        y, p = A.augment(x, np.random.default_rng(s), noises=[A.coloured_noise(np.random.default_rng(9), 8000)])
        assert np.isfinite(y).all() and np.max(np.abs(y)) <= 1.0
        assert 0.85 * 16000 < len(y) < 1.15 * 16000


def test_resample_tiny_and_empty():
    for n in (0, 1, 2, 3):
        y = A.resample(np.ones(n, np.float32), 44100, 16000)
        assert np.isfinite(y).all() and len(y) <= n
