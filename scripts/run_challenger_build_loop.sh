#!/bin/bash
set -u
HOST="srv-d6pi10c50q8c73dhp85g@ssh.oregon.render.com"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null"

for i in $(seq 1 40); do
  cnt=""
  for attempt in 1 2 3; do
    out=$($SSH "$HOST" 'cd /var/data/regbot-ch && git pull -q --ff-only 2>/dev/null; BUILD_TIME_BUDGET_SECONDS=40 CHALLENGER_CACHE_PATH=/var/data/challenger_embeddings_cache.json python scripts/build_challenger_embeddings.py 2>&1 | grep -oE "embedded_nodes\": [0-9]+"' 2>/dev/null)
    cnt=$(echo "$out" | grep -oE '[0-9]+' | tail -1)
    [ -n "$cnt" ] && break
    sleep 5
  done
  echo "iter $i: embedded=${cnt:-FAIL}"
  if [ "$cnt" = "6669" ]; then echo "BUILD_DONE"; break; fi
  sleep 2
done

echo "=== MEASURE ==="
$SSH "$HOST" 'cd /var/data/regbot-ch && CHALLENGER_CACHE_PATH=/var/data/challenger_embeddings_cache.json python scripts/measure_challenger_hybrid.py 2>&1 | tail -40' 2>/dev/null
