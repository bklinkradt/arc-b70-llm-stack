#!/usr/bin/env bash
# MTP draft-depth sweep on the second B70 (docker-compose.mtp.yml, 127.0.0.1:8002).
# Card 0 keeps serving chat. For each depth: decode at temperature 0 (code, prose),
# 4 concurrent requests, temperature 0.7, per-position draft acceptance from /metrics,
# repro.py's long-prompt corruption check, and the kernel log. Gives the card back to
# llama-swap at the end.
#
# Usage, from anywhere: qwen-vllm/mtp-depth.sh [out-dir]
# Env: DEPTHS (default "3 2 4"); REPRO_PLAN, the repro.py modes to run 5 rounds each
# (default "nocache cache"; the 40-round check in README.md is 6 nocache + 2 cache);
# MTP_IMAGE / MTP_CACHE to test another image (see docker-compose.mtp.yml).
set -uo pipefail
cd "$(dirname "$0")"

OUT=${1:-logs/$(date +%F)-mtp-depth}
DEPTHS=${DEPTHS:-"3 2 4"}
REPRO_PLAN=${REPRO_PLAN:-"nocache cache"}
URL=http://localhost:8002
mkdir -p "$OUT"

metrics() { curl -s "$URL/metrics" | grep -E '^vllm:spec_decode_num_(drafts|accepted_tokens_per_pos)_total'; }

acceptance() {  # before-file after-file
    python3 - "$1" "$2" <<'EOF'
import re, sys
def load(p):
    d = {}
    for line in open(p):
        m = re.match(r'(vllm:spec_decode_num_\w+?)_total\{([^}]*)\} (\S+)', line)
        if m:
            pos = re.search(r'position="(\d+)"', m[2])
            d[(m[1], pos[1] if pos else None)] = float(m[3])
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
drafts = b[("vllm:spec_decode_num_drafts", None)] - a.get(("vllm:spec_decode_num_drafts", None), 0)
pos = sorted(int(k[1]) for k in b if k[1] is not None)
rates = [(b[("vllm:spec_decode_num_accepted_tokens_per_pos", str(p))]
          - a.get(("vllm:spec_decode_num_accepted_tokens_per_pos", str(p)), 0)) / drafts for p in pos]
print("  acceptance by position: " + " ".join(f"{r:.0%}" for r in rates)
      + f"  mean tokens per step: {1 + sum(rates):.2f}")
EOF
}

bench() {  # label args...
    local label=$1; shift
    BENCH_URL=$URL python3 bench.py "$@" > "$dir/bench-$label.txt" 2>&1
    # After the shift, "$@" is bench.py's args: runs mode concurrency temperature.
    python3 - "$dir/bench-$label.txt" "$label" "$3" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
per = [float(x) for x in re.findall(r"decode\s+([\d.]+) tok/s", text)]
conc = int(sys.argv[3])
avg = re.search(r"avg decode: ([\d.]+)", text)
line = f"  {sys.argv[2]:14s} per request {avg[1] if avg else '?':>5s} tok/s"
if conc > 1 and per:
    line += f"   total {sum(per) / (len(per) / conc):.0f} tok/s"
if "warning" in text: line += "   (warning: other requests running)"
print(line)
EOF
}

echo "MTP depth sweep $(date '+%F %T'), image ${MTP_IMAGE:-$(grep -o 'vllm-openai-xpu:[^@]*' docker-compose.yml | head -1)}" | tee -a "$OUT/summary.txt"
for n in $DEPTHS; do
    dir="$OUT/mtp$n"; mkdir -p "$dir"
    since=$(date '+%F %T')
    {
        echo "== num_speculative_tokens=$n"
        make -s -C .. mtp MTP_N="$n" > "$dir/start.txt" 2>&1
        up=0
        for _ in $(seq 180); do  # first start compiles kernels; allow 30 min
            curl -sf "$URL/health" >/dev/null && { up=1; break; }
            docker inspect -f '{{.State.Running}}' qwen-vllm-mtp 2>/dev/null | grep -q true || break
            sleep 10
        done
        docker logs qwen-vllm-mtp > "$dir/server-start.log" 2>&1
        if [ $up = 0 ]; then
            echo "  server failed to start"; tail -5 "$dir/server-start.log" | sed 's/^/    /'
        else
            echo "  up after $(( $(date +%s) - $(date -d "$since" +%s) ))s"
            BENCH_URL=$URL python3 bench.py 1 code 1 0 > /dev/null 2>&1   # warm-up
            metrics > "$dir/metrics-before.txt"
            bench code-t0     5 code 1 0
            bench prose-t0    5 prose 1 0
            metrics > "$dir/metrics-after.txt"
            acceptance "$dir/metrics-before.txt" "$dir/metrics-after.txt"
            bench code-t0.7   5 code 1 0.7
            bench code-4x     3 code 4 0
            bench prose-4x    3 prose 4 0
            i=0
            for mode in $REPRO_PLAN; do
                i=$((i + 1))
                BENCH_URL=$URL python3 repro.py 5 $mode > "$dir/repro-$i-$mode.txt" 2>&1
                echo "  repro $i $mode: $(tail -1 "$dir/repro-$i-$mode.txt")"
                grep -E "CORRUPT|tail:|Error" "$dir/repro-$i-$mode.txt" | head -6 | sed 's/^/    /'
            done
            docker ps --format '{{.Names}}' | grep -qx qwen-vllm-mtp || echo "  ENGINE DIED during the run"
            docker logs qwen-vllm-mtp > "$dir/server.log" 2>&1
        fi
        f=$(journalctl -k --since "$since" | grep -iE 'xe .*(fault|reset|timed out|CAT error)')
        echo "  xe faults: ${f:-none}"
    } 2>&1 | tee -a "$OUT/summary.txt"
done

make -s -C .. mtp-down > /dev/null 2>&1
echo "card returned to llama-swap $(date '+%T')" | tee -a "$OUT/summary.txt"
