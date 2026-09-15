#!/usr/bin/env python3
"""Score the D37 case answers with a local multilingual NLI groundedness model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.groundedness import build_premise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="joeddav/xlm-roberta-large-xnli")
    args = parser.parse_args()

    evidence_units = json.loads(args.evidence.read_text(encoding="utf-8"))["evidence_units"]
    verbatim_by_id = {str(unit["evidence_id"]): str(unit["raw_text"]) for unit in evidence_units}
    run = json.loads(args.run.read_text(encoding="utf-8"))

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    entailment_index = next(
        index for index, label in model.config.id2label.items() if label == "entailment"
    )

    premises = []
    hypotheses = []
    records = run["records"]
    for record in records:
        retrieved_ids = [str(hit["evidence_id"]) for hit in record["retrieved_evidence"]]
        premise = build_premise(
            [
                {"raw_text": verbatim_by_id[evidence_id]}
                for evidence_id in retrieved_ids
                if evidence_id in verbatim_by_id
            ]
        )
        premises.append(premise)
        hypotheses.append(str(record["answer"]))

    inputs = tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)
    entailment_scores = probabilities[:, entailment_index]

    results = []
    for record, score in zip(records, entailment_scores):
        results.append(
            {
                "case_id": record["case"]["id"],
                "groundedness_score": round(float(score), 4),
                "gate_status": record["gate"]["status"],
            }
        )

    out = {
        "schema_version": "d37-groundedness/v1",
        "model": args.model,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
