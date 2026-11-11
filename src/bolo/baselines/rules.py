"""Rule / keyword baseline.

A hand-written cue-word system of the kind a developer would write in an afternoon. It matches on
phonetic keys (so spelling variants and Devanagari work) and extracts slots as the longest run of
non-function words. It never reads the slot-value pools, so test values are as unseen to it as to
the learned models. The only lexicon data it uses is number words (closed vocabulary).
"""

from __future__ import annotations

from ..config import Config
from ..normalize import Normalizer, phonetic_key


def _keys(words: str) -> set[str]:
    return {phonetic_key(w) for w in words.split()}


VOL = _keys("volume vol volumn awaaz awaz aawaz sound")
BRIGHT = _keys("brightness brightnes roshni roshani")
CLAUDE = _keys("claude")
TERM = _keys("terminal termnl terminl shell console cmd")
CALL = _keys("call cal kall ring dial")
OPEN = _keys("kholo khol kholna kholiye open start launch chalu chalao chala kolo khollo chahiye")
FILE_CUE = _keys("file folder fldr directory finder wali")
UNSUPPORTED = _keys("band bandh bnd close delete message msg bhejo install uninstall rename shutdown "
                    "restart lock screenshot wallpaper hatao mat banao badlo")
STATUS_WORDS = _keys("status batao btao dikhao check chk kitni kitna bacha bachi haal connected usage left")
TARGETS = {
    "battery": "battery", "batery": "battery", "battry": "battery", "charging": "battery", "charge": "battery",
    "wifi": "wifi", "internet": "internet", "net": "internet", "network": "internet",
    "storage": "storage", "disk": "storage", "space": "storage", "git": "git", "repo": "git",
    "cpu": "cpu", "processor": "cpu", "ram": "ram", "memory": "ram", "bluetooth": "bluetooth",
}
TARGET_KEYS = {phonetic_key(k): v for k, v in TARGETS.items()}
CALL_APPS = {"whatsapp": "whatsapp", "facetime": "facetime", "telegram": "telegram", "zoom": "zoom",
             "meet": "google meet", "teams": "teams", "skype": "skype", "discord": "discord", "signal": "signal"}
CALL_APP_KEYS = {phonetic_key(k): v for k, v in CALL_APPS.items()}
DIR_UP = _keys("badhao badha bdhao barhao tez zyada jyada increase up")
DIR_DOWN = _keys("kam km ghatao dheere dhire dheemi halki decrease down")
DIR_MUTE = _keys("mute band bandh")
DIR_UNMUTE = _keys("unmute wapas")
DIR_MAX = _keys("full max poora poori")
DIR_MIN = _keys("minimum sabse")
STOP = _keys(
    "yaar yar yr zara jara zra please plz pls plis bhai ek baar bar accha acha achha umm suno haan chalo abhi na "
    "jaldi hey kya kia aap mujhe muje mjhe mera meri mere can you could the a my me show to "
    "ka ki ke ko se pe par mein me mai mei main andar ander liye wale wala wali vale vala naya new "
    "project window line command code kaam karna kar kr karo kro karoo do de dijiye kijiye kijie dijie "
    "kholo khol kholna kholiye kolo khollo open start launch chalu chaalu chalao chala chahiye chaiye "
    "sakte ho hai he h hy aur or le bulao video number milao baat karao ring call cal kall dial phone fone "
    "lagao lgao laga lga status batao btao bata dikhao dkhao check chk usage kitna kitni bacha bachi "
    "haal chal raha rha connected on left much how is what's set level percent turn "
    "app folder fldr directory file finder in terminal shell console cmd claude kya ya give lo"
)


class RuleBaseline:
    name = "rules"

    def __init__(self, cfg: Config):
        self.norm = Normalizer.from_config(cfg)
        self.num_words: dict[str, int] = {}
        for v, (roman, _dev) in cfg.numbers_hi.items():
            self.num_words[phonetic_key(roman)] = v
        for v, word in cfg.numbers_en.items():
            if " " not in word:
                self.num_words[phonetic_key(word)] = v
        self.num_words[phonetic_key("twenty")] = 20

    def _entity(self, words: list[str], keys: list[str], extra_stop: set[str] = frozenset()) -> str | None:
        runs, cur = [], []
        for w, k in zip(words, keys):
            if k in STOP or k in extra_stop or k.isdigit():
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append(w)
        if cur:
            runs.append(cur)
        if not runs:
            return None
        return " ".join(max(runs, key=len))

    def _level(self, words: list[str], keys: list[str]) -> int | None:
        for i, (w, k) in enumerate(zip(words, keys)):
            if w.isdigit():
                return max(0, min(100, int(w)))
            if k in self.num_words:
                # "twenty five" / "seventy five"
                if i + 1 < len(keys) and keys[i + 1] == phonetic_key("five") and k in (phonetic_key("twenty"), phonetic_key("seventy")):
                    return self.num_words[k] + 5
                return self.num_words[k]
        return None

    def _direction(self, keys: list[str], allowed: set[str]) -> str | None:
        ks = set(keys)
        if "min" in allowed and (ks & DIR_MIN):
            return "min"
        if "unmute" in allowed and (ks & DIR_UNMUTE):
            return "unmute"
        if "mute" in allowed and (ks & DIR_MUTE):
            return "mute"
        if ks & DIR_MAX:
            return "max"
        if ks & DIR_DOWN:
            return "down"
        if ks & DIR_UP:
            return "up"
        return None

    def predict_one(self, text: str) -> dict:
        words = self.norm.roman_words(text)
        keys = [phonetic_key(w) for w in words]
        ks = set(keys)
        none = {"intent": "none", "slots": {}, "confidence": 1.0}
        if not words:
            return none
        if ks & VOL:
            lvl = self._level(words, keys)
            if lvl is not None:
                return {"intent": "set_volume", "slots": {"level": lvl}, "confidence": 1.0}
            d = self._direction(keys, {"up", "down", "mute", "unmute", "max"})
            return {"intent": "set_volume", "slots": {"direction": d}, "confidence": 1.0} if d else none
        if ks & BRIGHT or (phonetic_key("screen") in ks and not ks & UNSUPPORTED and (ks & (DIR_UP | DIR_DOWN | DIR_MAX))):
            lvl = self._level(words, keys)
            if lvl is not None:
                return {"intent": "set_brightness", "slots": {"level": lvl}, "confidence": 1.0}
            d = self._direction(keys, {"up", "down", "max", "min"})
            return {"intent": "set_brightness", "slots": {"direction": d}, "confidence": 1.0} if d else none
        if ks & UNSUPPORTED:
            return none
        if phonetic_key("on") in ks and ks & {phonetic_key("wifi"), phonetic_key("bluetooth")} and ks & {phonetic_key("karo"), phonetic_key("kro")}:
            return none
        if ks & CLAUDE:
            if ks & TERM:
                return none  # "terminal ya claude kholo"
            folder = self._entity(words, keys)
            return {"intent": "open_claude_in_folder", "slots": {"folder": folder} if folder else {}, "confidence": 1.0}
        if ks & TERM or (phonetic_key("command") in ks and phonetic_key("line") in ks):
            folder = self._entity(words, keys)
            return {"intent": "open_terminal", "slots": {"folder": folder} if folder else {}, "confidence": 1.0}
        is_call = bool(ks & CALL) or (phonetic_key("baat") in ks and phonetic_key("karao") in ks) or \
            (phonetic_key("number") in ks and phonetic_key("milao") in ks) or \
            (phonetic_key("phone") in ks and ks & _keys("karo kro lagao laga kar"))
        if is_call:
            app = next((CALL_APP_KEYS[k] for k in keys if k in CALL_APP_KEYS), None)
            if app is None and phonetic_key("phone") in ks:
                i = keys.index(phonetic_key("phone"))
                if i + 1 < len(keys) and keys[i + 1] in _keys("pe par se on"):
                    app = "phone"
            name = self._entity(words, keys, extra_stop=set(CALL_APP_KEYS) | _keys("google microsoft"))
            if not name:
                return none
            slots = {"name": name}
            if app:
                slots["app"] = app
            return {"intent": "call_contact", "slots": slots, "confidence": 1.0}
        targets = [TARGET_KEYS[k] for k in keys if k in TARGET_KEYS]
        if targets and (ks & STATUS_WORDS or phonetic_key("kya") in ks or len(words) <= 3):
            return {"intent": "check_status", "slots": {"target": targets[0]}, "confidence": 1.0}
        if ks & OPEN or phonetic_key("dikhao") in ks:
            ent = self._entity(words, keys)
            if not ent:
                return none
            if ks & FILE_CUE or phonetic_key("dikhao") in ks:
                return {"intent": "open_file_or_folder", "slots": {"path": ent}, "confidence": 1.0}
            return {"intent": "open_app", "slots": {"app": ent}, "confidence": 1.0}
        return none

    def predict(self, texts: list[str]) -> list[dict]:
        return [self.predict_one(t) for t in texts]
