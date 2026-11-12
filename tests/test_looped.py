from bolo.pipeline import looped


def test_looped():
    assert looped("cpu cpu cpu cpu cpu cpu")
    assert looped("volume kam karo volume kam karo volume kam karo")
    for ok in ("volume kam karo", "bhai bhai terminal kholo", "rahul ko call karo na yaar abhi please",
               "chalo 30 ghante mein remind karna ki notes padhne hain"):
        assert not looped(ok)
