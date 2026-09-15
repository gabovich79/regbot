#!/usr/bin/env python3
"""Build the D37 local vector index using an Ollama embedding model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.local_evidence_index import build_local_evidence_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="nomic-embed-text:v1.5")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/embed")
    args = parser.parse_args()

    artifact = json.loads(args.evidence.read_text(encoding="utf-8"))
    units = artifact["evidence_units"]

    def embed_many(texts: list[str]) -> list[list[float]]:
        request = Request(
            args.ollama_url,
            data=json.dumps({"model": args.model, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama /api/embed did not return an embeddings list")
        return embeddings

    index = build_local_evidence_index(units, model=args.model, embed_many=embed_many)
    index["source"] = artifact["source"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "model": args.model, "dimension": index["dimension"], "entries": len(index["entries"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
