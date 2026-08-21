"""Create a read-only JSON inventory of the current RegBot corpus.

Run locally or from a Render shell:
    python scripts/audit_corpus.py --db /opt/render/project/src/data/regbot.db
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import DB_PATH
from services.corpus_audit_service import audit_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RegBot document extraction and indexing")
    parser.add_argument("--db", default=DB_PATH, help="Path to regbot.db")
    parser.add_argument("--output", help="Write JSON report to this path instead of stdout")
    args = parser.parse_args()

    report = audit_corpus(args.db)
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(report_json + "\n", encoding="utf-8")
        print(f"Wrote corpus audit to {args.output}")
    else:
        print(report_json)


if __name__ == "__main__":
    main()
