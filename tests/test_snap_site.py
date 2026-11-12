from bolo.grounding import snap_site

SITES = ["moneycontrol", "youtube", "linkedin", "github", "stack overflow", "times of india"]


def test_split_name_joined():
    assert snap_site("money control", "open money control website".split(), SITES) == "moneycontrol"


def test_partial_span_extended_with_neighbours():
    w = "brave browser pe money control website".split()
    assert snap_site("control", w, SITES) == "moneycontrol"
    assert snap_site("money", w, SITES) == "moneycontrol"


def test_known_multiword_kept():
    assert snap_site("stack overflow", "stack overflow kholo".split(), SITES) == "stack overflow"


def test_unknown_site_passes_through():
    assert snap_site("example", "open example dot org website".split(), SITES) == "example"


def test_does_not_grab_unrelated_words():
    # "git" alone must not pull in a far-away word to become something else
    assert snap_site("tata", "open tata website".split(), SITES) == "tata"
