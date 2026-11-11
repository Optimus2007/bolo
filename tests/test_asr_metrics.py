from bolo.audio.metrics import corpus_scores, edit_distance, norm_words


def test_edit_distance():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], ["a"]) == 1


def test_norm_and_scores():
    assert norm_words("Terminal Kholo!") == ["terminal", "kholo"]
    s = corpus_scores(["terminal kholo"], ["terminal kholo"])
    assert s["wer"] == 0 and s["sentence_exact"] == 1
    s = corpus_scores(["terminal kholo"], ["terminal kolo"])
    assert s["wer"] > 0 and s["phonetic_wer"] == 0
    s = corpus_scores(["band karo"], ["बंद करो"])  # Devanagari hyp is transliterated
    assert s["phonetic_wer"] == 0, s
