#!/usr/bin/env bash
# Run the hierarchical challenger locally on this Mac.
# The OpenAI key is read silently into this process only; it is never saved.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

if ! "$PYTHON" -c 'import openai, numpy, tiktoken' >/dev/null 2>&1; then
  echo "Creating a local Python environment and installing project dependencies..."
  python3 -m venv .venv
  PYTHON=.venv/bin/python
  "$PYTHON" -m pip install -q -r requirements.txt
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -r -s -p "OpenAI API key (input hidden): " OPENAI_API_KEY
  echo
  export OPENAI_API_KEY
fi

CACHE="$ROOT/results/challenger_embeddings_cache_local.json"
export CHALLENGER_CACHE_PATH="$CACHE"

printf 'Building the local challenger cache...\n'
"$PYTHON" scripts/build_challenger_embeddings.py
printf '\nMeasuring hybrid retrieval...\n'
"$PYTHON" scripts/measure_challenger_hybrid.py
printf '\nFull results:\n'
"$PYTHON" -m json.tool results/challenger_hybrid_metrics.json
