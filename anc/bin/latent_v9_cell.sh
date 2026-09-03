#!/usr/bin/env bash
# Progressive-combination cell runner with GPU/CPU/VRAM utilization capture.
# Runs ON TLN HOST (not inside a container): each cell is one docker call
# for the runner + one for the frozen attacker, with nvidia-smi/vmstat
# sampled on both nodes for the full cell window.
#
# usage: latent_v9_cell.sh NAME MODEL SPLIT RESUME LATENT_DIM NOISE CHAFF
#          SEQ_LEN CLOUD_KIND EXPERTS LAYERS PORT GRAM STEPS [TOKEN_GAUGE=1]
#          [CHANNELS=1]
set -euo pipefail

NAME="$1"; MODEL="$2"; SPLIT="$3"; RESUME="$4"; LDIM="$5"; NOISE="$6"
CHAFF="$7"; SEQ="$8"; KIND="$9"; EXPERTS="${10}"; LAYERS="${11}"
PORT="${12}"; GRAM="${13}"; STEPS="${14}"; TOKEN_GAUGE="${15:-1}"
CHANNELS="${16:-1}"; PUBLIC_STEPS="${17:-0}"; URLS="${18:-}"

GAUGE_FLAG=()
if [ "$TOKEN_GAUGE" = "1" ]; then
  GAUGE_FLAG=(--secret-token-gauge)
fi
PUBLIC_FLAG=()
if [ "$PUBLIC_STEPS" != "0" ]; then
  PUBLIC_FLAG=(--public-corpus /workspace/experiments/models/wikitext2_public.txt
               --public-steps "$PUBLIC_STEPS")
fi

OUTDIR_HOST=$HOME/experiments/results/training
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
UTILD=$OUTDIR_HOST/util
mkdir -p "$UTILD" "$OUTDIR_HOST/bundles"
TS0=$(date +%s)

# start monitors (tln GPU+CPU, ucn GPU+CPU)
# GB10 unified memory: nvidia-smi reports utilization.gpu but memory.used is
# [N/A] on this platform, so memory footprint is sampled via docker stats
# (RSS == VRAM on unified memory) for the runner container and the sum of
# ucn's latent-cloud servers.
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader \
  -l 2 > "$UTILD/${NAME}_tln_gpu.log" 2>/dev/null &
M_GPU_T=$!
vmstat 2 > "$UTILD/${NAME}_tln_cpu.log" 2>/dev/null &
M_CPU_T=$!
ssh ucn 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -l 2' \
  > "$UTILD/${NAME}_ucn_gpu.log" 2>/dev/null &
M_GPU_K=$!
ssh ucn 'vmstat 2' > "$UTILD/${NAME}_ucn_cpu.log" 2>/dev/null &
M_CPU_K=$!
( while true; do docker stats --no-stream --format "{{.MemUsage}}" \
    "pxrun_${NAME}" 2>/dev/null; sleep 2; done \
  > "$UTILD/${NAME}_tln_mem.log" ) &
M_MEM_T=$!
( ssh ucn 'while true; do docker stats --no-stream --format "{{.MemUsage}}" $(docker ps --filter name=latent-cloud --format "{{.Names}}" | tr "\n" " "); sleep 2; done' \
  > "$UTILD/${NAME}_ucn_mem.log" 2>/dev/null ) &
M_MEM_K=$!

stop_monitors () {
  kill $M_GPU_T $M_CPU_T $M_GPU_K $M_CPU_K $M_MEM_T $M_MEM_K 2>/dev/null
  wait $M_GPU_T $M_CPU_T $M_GPU_K $M_CPU_K $M_MEM_T $M_MEM_K 2>/dev/null
  :
}

echo "=== cell $NAME (model=$(basename $MODEL) split=$SPLIT/$RESUME d=$LDIM n=$NOISE chaff=$CHAFF seq=$SEQ kind=$KIND E=$EXPERTS L=$LAYERS port=$PORT gram=$GRAM steps=$STEPS) ==="

docker run --rm --name "pxrun_${NAME}" --gpus all --network host --ipc host \
  -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -v $HOME/experiments:/workspace/experiments \
  -v $HOME/dtraining:$HOME/dtraining -w $HOME/dtraining \
  split-inference:spark \
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus /workspace/experiments/models/wikitext2_corpus.txt \
    --output "$OUTDIR/latent_v9_${NAME}.json" \
    --cloud-url "wss://ucn:$PORT" --cloud-tls-ca "$CA" \
    ${URLS:+--cloud-urls "$URLS"} \
    --cloud-kind "$KIND" --cloud-experts "$EXPERTS" --cloud-layers "$LAYERS" \
    --cloud-channels "$CHANNELS" \
    --secret-wire-rotation --secret-token-permutation "${GAUGE_FLAG[@]}" \
    --latent-dim "$LDIM" --noise-multiplier "$NOISE" --clip-norm 1.0 \
    --split-after "$SPLIT" --resume-after "$RESUME" --seq-len "$SEQ" \
    --steps "$STEPS" --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --adversary-strength 1.0 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens "$CHAFF" --gram-flatten "$GRAM" \
    "${PUBLIC_FLAG[@]}" --seed 42 \
    --attacker-bundle "$BUNDLEDIR/latent_v9_${NAME}_bundle.pt"
RC=$?

if [ $RC -eq 0 ]; then
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -v $HOME/experiments:/workspace/experiments \
    -v $HOME/dtraining:$HOME/dtraining -w $HOME/dtraining \
    split-inference:spark \
    python3 -m attacker --attack latent-probe \
      --bundle "$BUNDLEDIR/latent_v9_${NAME}_bundle.pt" \
      --output "$OUTDIR/latent_v9_${NAME}_attacker.json"
  RC=$?
fi
rm -f "$OUTDIR_HOST/bundles/latent_v9_${NAME}_bundle.pt"

stop_monitors
TS1=$(date +%s)

# summarize utilization
python3 - "$NAME" $((TS1-TS0)) <<'EOF'
import json, sys
name, wall = sys.argv[1], int(sys.argv[2])
util = {}

def gpu_log(path):
    utils = []
    try:
        for line in open(path):
            value = line.split(",")[0].strip().rstrip(" %")
            if value and value != "[N/A]" and value.replace(".", "").isdigit():
                utils.append(float(value))
    except FileNotFoundError:
        pass
    return ({"gpu_util_mean_pct": round(sum(utils)/len(utils), 1),
             "gpu_util_peak_pct": max(utils)} if utils else {})

def cpu_log(path):
    # Parse vmstat: locate the "id" (idle) column from the header row.
    idles = []
    try:
        idx = None
        for line in open(path):
            cols = line.split()
            if "id" in cols and idx is None:
                idx = cols.index("id")
                continue
            if idx is not None and len(cols) > idx and cols[idx].isdigit():
                idles.append(float(cols[idx]))
    except FileNotFoundError:
        pass
    return {"cpu_util_mean_pct": round(100 - sum(idles)/len(idles), 1),
            "cpu_util_peak_pct": round(100 - min(idles), 1)} if idles else {}

def mem_log(path):
    # docker stats MemUsage lines like "1.234GiB / 128GiB" or "512MiB / ..."
    peak = 0.0
    try:
        for line in open(path):
            used = line.split("/")[0].strip()
            if used.endswith("GiB"):
                peak = max(peak, float(used[:-3]) * 1024)
            elif used.endswith("MiB"):
                peak = max(peak, float(used[:-3]))
            elif used.endswith("KiB"):
                peak = max(peak, float(used[:-3]) / 1024)
    except FileNotFoundError:
        pass
    return {"mem_peak_mib": round(peak, 1)} if peak else {}

util["tln_gpu"] = gpu_log(f"/home/geo/experiments/results/training/util/{name}_tln_gpu.log")
util["ucn_gpu"] = gpu_log(f"/home/geo/experiments/results/training/util/{name}_ucn_gpu.log")
util["tln_cpu"] = cpu_log(f"/home/geo/experiments/results/training/util/{name}_tln_cpu.log")
util["ucn_cpu"] = cpu_log(f"/home/geo/experiments/results/training/util/{name}_ucn_cpu.log")
util["tln_mem"] = mem_log(f"/home/geo/experiments/results/training/util/{name}_tln_mem.log")
util["ucn_mem"] = mem_log(f"/home/geo/experiments/results/training/util/{name}_ucn_mem.log")
util["wall_seconds"] = wall
json.dump(util, open(
    f"/home/geo/experiments/results/training/latent_v9_{name}_util.json", "w"),
    indent=1)
print("util:", json.dumps(util))
EOF

[ $RC -eq 0 ] && echo "=== cell $NAME done ===" || echo "=== cell $NAME FAILED rc=$RC ==="
