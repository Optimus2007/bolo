from bolo.grounding import Environment, ground, similarity

ENV = Environment(folders=["kaggle", "ml notes", "bolo"], files=["resume pdf"],
                  apps=["bitwarden", "vs code"], contacts=["sakshi", "rahul sharma", "mummy"])


def test_similarity_handles_variants():
    assert similarity("kaigal", "kaggle") > 0.72
    assert similarity("sakshee", "sakshi") == 1.0
    assert similarity("chrome", "bolo") < 0.5


def test_snaps_transliterated_values():
    g = ground({"intent": "open_claude_in_folder", "slots": {"folder": "kaigal"}}, ENV)
    assert g["slots"] == {"folder": "kaggle"}


def test_switches_app_vs_folder_by_what_exists():
    g = ground({"intent": "open_file_or_folder", "slots": {"path": "bitwarden"}}, ENV)
    assert g["intent"] == "open_app" and g["slots"] == {"app": "bitwarden"}


def test_abstains_when_entity_missing():
    g = ground({"intent": "call_contact", "slots": {"name": "zorawar"}}, ENV)
    assert g["intent"] == "none" and "abstain" in g["grounding"]


def test_closed_slots_untouched():
    p = {"intent": "set_volume", "slots": {"level": 40}}
    assert ground(p, ENV)["slots"] == {"level": 40}



def test_unknown_kind_passes_through_unverified():
    from bolo.grounding import Environment, ground

    env = Environment(folders=["photos"], files=[], apps=["vlc"], contacts=[])  # contacts not shared
    g = ground({"intent": "call_contact", "slots": {"name": "diya"}, "confidence": 0.9}, env)
    assert g["intent"] == "call_contact" and g["slots"]["name"] == "diya"
    assert g["grounding"]["name"]["verified"] is False
    g2 = ground({"intent": "open_app", "slots": {"app": "bitword done"}, "confidence": 0.9}, env)
    assert g2["intent"] == "none"  # apps ARE known, and nothing matches: still abstain
