"""Validate {intent, slots} labels against the command schema."""

from __future__ import annotations


class LabelError(ValueError):
    pass


def validate_label(schema: dict, intent: str, slots: dict) -> None:
    """Raise LabelError if (intent, slots) violates the schema."""
    intents = schema["intents"]
    if intent not in intents:
        raise LabelError(f"unknown intent {intent!r}")
    spec = intents[intent]
    allowed = spec.get("slots") or {}
    for name, value in slots.items():
        if name not in allowed:
            raise LabelError(f"slot {name!r} not allowed for {intent}")
        stype = schema["slot_types"][allowed[name]["type"]]
        if allowed[name]["type"] in ("level", "amount"):
            lo = 0 if allowed[name]["type"] == "level" else 1
            if not isinstance(value, int) or not lo <= value <= 100:
                raise LabelError(f"{name} must be int {lo}-100, got {value!r}")
        elif "values" in stype and value not in stype["values"]:
            raise LabelError(f"{value!r} not a valid {allowed[name]['type']}")
        elif not isinstance(value, (str, int)) or value == "":
            raise LabelError(f"bad value for {name}: {value!r}")
    for name, sdef in allowed.items():
        if sdef.get("required") and name not in slots:
            raise LabelError(f"{intent} requires slot {name!r}")
    one_of = spec.get("one_of")
    if one_of and sum(1 for s in one_of if s in slots) != 1:
        raise LabelError(f"{intent} needs exactly one of {one_of}")
