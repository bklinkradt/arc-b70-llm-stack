#!/usr/bin/env bash
# Garbled-output check on the second B70: Vulkan (Mesa 26.2) as the control, SYCL with
# GGML_SYCL_ENABLE_OPT on (default) and off (the reported workaround). For each quant and
# backend: short prompts, then ../qwen-vllm/repro.py's ~21k-token multi-turn rounds with
# and without prefix caching, then the kernel log. Outputs land in $OUT so the SYCL text
# can be diffed against Vulkan's at temperature 0.
#
# Usage (card already borrowed with `make take-card`):
#   ./garble-test.sh [out-dir]
# Env: QUANTS (GGUF paths under $MODELS_DIR, default ~/models), CONFIGS, ROUNDS.
set -uo pipefail
cd "$(dirname "$0")"

OUT=${1:-logs/$(date +%F)-gemma-garble}
QUANTS=${QUANTS:-"gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_0.gguf gemma-4-12B-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf"}
CONFIGS=${CONFIGS:-"vulkan-mesa26 sycl-opt1 sycl-opt0"}
ROUNDS=${ROUNDS:-5}
ALIAS=gemma-4-12B
mkdir -p "$OUT"

port_of() { case $1 in vulkan-mesa26) echo 8093;; sycl-*) echo 8091;; esac; }
container_of() { case $1 in vulkan-mesa26) echo llamacpp-vulkan-mesa26;; sycl-*) echo llamacpp-sycl;; esac; }

start() {  # config model
    local cfg=$1 model=$2
    case $cfg in
        vulkan-mesa26) make -s vulkan-mesa26 MODEL="$model" ALIAS=$ALIAS ;;
        sycl-opt1)     make -s sycl MODEL="$model" ALIAS=$ALIAS SYCL_OPT=1 ;;
        sycl-opt0)     make -s sycl MODEL="$model" ALIAS=$ALIAS SYCL_OPT=0 ;;
    esac >/dev/null 2>&1
    for _ in $(seq 180); do
        curl -sf "localhost:$(port_of "$cfg")/health" >/dev/null && return 0
        docker inspect -f '{{.State.Running}}' "$(container_of "$cfg")" 2>/dev/null | grep -q true || break
        sleep 2
    done
    echo "  server failed to start"; docker logs "$(container_of "$cfg")" 2>&1 | tail -5
    return 1
}

short_prompts() {  # url outdir
    BENCH_URL=$1 SHORT_OUT=$2 python3 - <<'EOF'
import importlib.util, json, os, sys, urllib.request
sys.argv = ["repro.py"]
spec = importlib.util.spec_from_file_location("repro", "../qwen-vllm/repro.py")
repro = importlib.util.module_from_spec(spec); spec.loader.exec_module(repro)
PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a bash script that finds the ten largest files under a directory and prints their sizes in human-readable form.",
    "Summarize the causes of the French Revolution in about 200 words.",
    "A train leaves at 14:05 and travels 312 km at an average of 96 km/h. When does it arrive? Show your working.",
    "Write a SQL query that returns each customer's three most recent orders, then explain how it works.",
    "Translate into German, then explain two grammar points in English: 'The weather was bad, so we stayed at home and read books.'",
]
out, bad = os.environ["SHORT_OUT"], 0
for i, p in enumerate(PROMPTS, 1):
    body = json.dumps({"model": repro.MODEL, "messages": [{"role": "user", "content": p}],
                       "max_tokens": 600, "temperature": 0.0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(repro.URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r)["choices"][0]["message"]["content"] or ""
    open(f"{out}/short-{i}.txt", "w").write(text)
    # Short answers can legitimately end under repro.py's 400-char floor; pad past it.
    c = repro.corrupt(text.ljust(400))
    bad += c
    print(f"  short {i}: {len(text):5d} chars  {'CORRUPT' if c else 'ok'}")
    if c: print("     tail:", repr(text[-200:]))
print(f"  short corrupted: {bad}/{len(PROMPTS)}")
EOF
}

# Appends, so a run split over several invocations (QUANTS / CONFIGS subsets) shares one summary.
echo "Gemma garble test $(date '+%F %T'), llama.cpp fc07d781e" | tee -a "$OUT/summary.txt"
for model in $QUANTS; do
    q=$(basename "$model" .gguf)
    for cfg in $CONFIGS; do
        dir="$OUT/$q/$cfg"; mkdir -p "$dir"
        since=$(date '+%F %T')
        {
            echo "== $q  $cfg"
            if start "$cfg" "$model"; then
                url="http://localhost:$(port_of "$cfg")"
                docker logs "$(container_of "$cfg")" 2>&1 | grep -E "GGML_SYCL_ENABLE_OPT|ggml_vulkan: 0 =" | head -1 | sed 's/^/  /'
                short_prompts "$url" "$dir"
                for mode in nocache cache; do
                    BENCH_URL=$url BENCH_MODEL=$ALIAS python3 ../qwen-vllm/repro.py "$ROUNDS" $mode \
                        > "$dir/repro-$mode.txt" 2>&1
                    echo "  repro $mode: $(tail -1 "$dir/repro-$mode.txt")"
                    grep -E "CORRUPT|tail:|Error|error" "$dir/repro-$mode.txt" | head -6 | sed 's/^/    /'
                done
                docker logs "$(container_of "$cfg")" > "$dir/server.log" 2>&1
            fi
            f=$(journalctl -k --since "$since" | grep -iE 'xe .*(fault|reset|timed out|CAT error)')
            echo "  xe faults: ${f:-none}"
        } 2>&1 | tee -a "$OUT/summary.txt"
    done
done

# Compare each SYCL short answer with Vulkan's. At temperature 0 small numeric
# differences can still change a token and send the text down another path, so a low
# ratio is a pointer to read the pair, not a verdict.
echo "== similarity to vulkan-mesa26 (short prompts, difflib ratio)" | tee -a "$OUT/summary.txt"
python3 - "$OUT" <<'EOF' | tee -a "$OUT/summary.txt"
import difflib, pathlib, sys
root = pathlib.Path(sys.argv[1])
for qd in sorted(p for p in root.iterdir() if p.is_dir()):
    ref = qd / "vulkan-mesa26"
    for cfg in ("sycl-opt1", "sycl-opt0"):
        if not (qd / cfg).exists() or not ref.exists(): continue
        r = [difflib.SequenceMatcher(None, (ref / f.name).read_text(), f.read_text()).ratio()
             for f in sorted((qd / cfg).glob("short-*.txt")) if (ref / f.name).exists()]
        print(f"  {qd.name:28s} {cfg:10s} " + " ".join(f"{x:.2f}" for x in r))
EOF
