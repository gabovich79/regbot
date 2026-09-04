#!/usr/bin/env bash
# Run the hierarchical challenger locally on this Mac.
# The OpenAI key is read silently into this process only; it is never saved.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer the known-good Python 3.11 project environment. The local .venv may
# have been created with Python 3.14, for which this project's pinned PyMuPDF
# lacks a prebuilt wheel and attempts an unnecessary source compilation.
CANDIDATES=()
[[ -n "${REGBOT_PYTHON:-}" ]] && CANDIDATES+=("$REGBOT_PYTHON")
CANDIDATES+=(
  "$HOME/.hermes/hermes-agent/venv/bin/python"
  "/tmp/regbot-clean/.venv/bin/python"
  "$ROOT/.venv/bin/python"
)
PYTHON=""
for candidate in "${CANDIDATES[@]}"; do
  if [[ -x "$candidate" ]] && "$candidate" -c 'import fitz, openai, numpy, tiktoken' >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  cat >&2 <<'EOF'
No compatible project Python environment was found.
Use Python 3.11 (not 3.14) and install requirements once, then rerun:
  /opt/homebrew/bin/python3.11 -m venv /tmp/regbot-clean/.venv
  /tmp/regbot-clean/.venv/bin/python -m pip install -r requirements.txt
EOF
  exit 2
fi

if [[ "${1:-}" == "--preflight" ]]; then
  echo "Python: $($PYTHON --version)"
  echo "PyMuPDF: OK"
  echo "Runner Python: $PYTHON"
  exit 0
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
printf '\nRunning candidate-fusion ablations...\n'
"$PYTHON" scripts/run_challenger_ablations.py
printf '\nFull ablation results:\n'
"$PYTHON" -m json.tool results/challenger_ablation_results.json
printf '\nRunning promotion gate (tuning + held-out + legacy)...\n'
"$PYTHON" scripts/measure_challenger_gate.py
