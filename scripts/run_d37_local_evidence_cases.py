#!/usr/bin/env python3
"""Run pending D37 evidence cases through local retrieval, synthesis, and gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.d37_evidence_answer_service import (
    build_grounded_prompt,
    gate_grounded_answer,
    resolve_citations,
)
from services.local_evidence_index import search_local_evidence_index
from services.local_ollama_client import build_constrained_chat_payload


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", default="nomic-embed-text:v1.5")
    parser.add_argument("--chat-model", default="qwen3.5:9b-ctx64k")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    index = json.loads(args.index.read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]

    def embed_query(question: str) -> list[float]:
        payload = post_json(
            f"{args.ollama_base_url}/api/embed",
            {"model": args.embedding_model, "input": question},
        )
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            raise RuntimeError("Ollama query embedding response is invalid")
        return embeddings[0]

    records = []
    for case in cases:
        hits = search_local_evidence_index(
            case["question"], index, embed_query=embed_query, top_k=len(index["entries"])
        )
        prompt = build_grounded_prompt(case["question"], hits)
        chat = post_json(
            f"{args.ollama_base_url}/api/chat",
            build_constrained_chat_payload(model=args.chat_model, prompt=prompt),
        )
        answer = str(chat.get("message", {}).get("content") or "")
        gate = gate_grounded_answer(answer, hits)
        resolved = resolve_citations(answer, hits)
        expected_ids = set(case["expected_evidence_ids"])
        retrieved_ids = {hit["evidence_id"] for hit in hits}
        records.append(
            {
                "case": case,
                "retrieved_evidence": [
                    {
                        "evidence_id": hit["evidence_id"],
                        "section": hit["section"],
                        "page_start": hit["page_start"],
                        "page_end": hit["page_end"],
                        "score": hit["score"],
                    }
                    for hit in hits
                ],
                "retrieval_expected_coverage": expected_ids.issubset(retrieved_ids),
                "answer": answer,
                "gate": gate,
                "resolved_citations": resolved["citations"],
                "unresolved_ids": resolved["unresolved_ids"],
            }
        )

    result = {
        "schema_version": "d37-local-evidence-run/v1",
        "models": {"embedding": args.embedding_model, "chat": args.chat_model},
        "case_count": len(records),
        "all_retrieval_expected_coverage": all(record["retrieval_expected_coverage"] for record in records),
        "all_gate_pass": all(record["gate"]["status"] == "pass" for record in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("case_count", "all_retrieval_expected_coverage", "all_gate_pass")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
