"""§9.6: join the exported labels with the hidden key.

    python3 -m analysis.messages.coding.importer ~/Downloads/labels_xxxxxxxx.json

Refuses labels made for another item file and incomplete coding. Writes
coding/sample/coded.csv, one row per item with its label and key fields.
"""
import csv
import hashlib
import json
import os
import sys

from analysis.messages.coding.sampler import ITEMS, KEY, OUT

CODED = os.path.join(OUT, "coded.csv")
FIELDS = ("id", "label", "extractor", "tier", "truth", "episode", "ply", "letter", "cell", "proto", "text")


def join(items, key, labels):
    """Rows of items x key x label; raises ValueError on a missing or unknown label."""
    ids = {it["id"] for it in items}
    missing, unknown = sorted(ids - set(labels)), sorted(set(labels) - ids)
    bad = sorted(k for k, v in labels.items() if v not in ("yes", "no"))
    if missing or unknown or bad:
        raise ValueError(f"missing {len(missing)} {missing[:5]}, unknown {unknown[:5]}, bad values {bad[:5]}")
    return [dict(id=it["id"], label=labels[it["id"]], text=it["text"], **key[it["id"]]) for it in items]


def main(argv):
    raw = open(ITEMS, "rb").read()
    lab = json.load(open(argv[0]))
    if lab["items_sha256"] != hashlib.sha256(raw).hexdigest():
        raise SystemExit("refused: these labels were made for a different item file")
    rows = join(json.loads(raw), json.load(open(KEY))["key"], lab["labels"])
    with open(CODED, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} coded items to {CODED}")


if __name__ == "__main__":
    main(sys.argv[1:])
