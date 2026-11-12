"""Re-evaluate saved Bolo Tiny models with the current threshold rule (no retraining) and save the threshold
next to each model so the pipeline uses the same cut-off that the reported numbers were measured with.

  .venv/bin/python scripts/rescore_tiny.py phono_views_aug 0 1 2
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bolo.config import load_config  # noqa: E402
from bolo.evaluate import load_rows, run  # noqa: E402
from bolo.models.tiny import BoloTiny  # noqa: E402

name, seeds = sys.argv[1], sys.argv[2:]
data = "data/text/v1"
cfg = load_config()
dev, test = load_rows(f"{data}/dev.jsonl"), load_rows(f"{data}/test.jsonl")
for s in seeds:
    d = Path(f"models/bolo-tiny/{name}-s{s}")
    m = BoloTiny.load(cfg, str(d))
    res = run(m, cfg.schema, test, dev_rows=dev)
    (d / "threshold.json").write_text(json.dumps({"threshold": res["threshold"], "rule": "best dev EM with dev FAR <= 3%"}))
    rp = Path(f"results/text/tiny_{name}-s{s}.json")
    old = json.loads(rp.read_text()) if rp.exists() else {}
    old.update(res)
    rp.write_text(json.dumps(old, indent=1, ensure_ascii=False))
    t = res["test"]
    print(f"s{s} thr {res['threshold']} | EM {t['exact_match']:.3f} | intent {t['intent_acc']:.3f} | "
          f"FAR {t['false_action_rate']:.3f} | over-abstain {t['over_abstain_rate']:.3f}")
