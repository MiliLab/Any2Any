#!/usr/bin/env bash
# Any2Any Multi-Direction Training Script
# Runs DiT-S/4 training on 14 translation directions sequentially (8 GPUs per direction).
#
# Direction ID mapping:
#   1  Opt-Sar   opt->sar      2  Opt-Sar   sar->opt
#   3  Opt-Pan   opt->pan      4  Opt-Pan   pan->opt
#   5  Opt-Ms    opt->ms       6  Opt-Ms    ms->opt
#   7  Opt-Nir   opt->nir      8  Opt-Nir   nir->opt
#   9  Nir-MS    nir->ms      10  Nir-MS    ms->nir
#  11  Sar-Ms    sar->ms      12  Sar-Ms    ms->sar
#  13  Sar-Nir   sar->nir     14  Sar-Nir   nir->sar
#
# Usage:
#   # Run all 14 directions:
#   bash train.sh
#
#   # Run specific directions:
#   export SERVER_RUN_DIRECTIONS="1,2,3,4"
#   bash train.sh
#
# Environment variables:
#   DIT_ROOT              Project root (default: current directory)
#   MANIFEST_BASE_DIR     Base dir for manifest paths (default: current directory)
#   RSUT_JSON_TRAIN_DIR   Training JSON directory
#   RSUT_JSON_VALID_DIR   Validation JSON directory
#   RSUT_JSON_TEST_DIR    Test JSON directory
#   AE_CKPT_SAR/OPT/MS/NIR/PAN  VAE checkpoint paths
#   MAX_TRAIN_STEPS       Steps per direction (default: 50000)
#   CKPT_EVERY            Checkpoint interval (default: 2500)
#   EVAL_EVERY            Evaluation interval (default: 10000)
#   GLOBAL_BATCH_SIZE     Batch size (default: 384)
#   NUM_WORKERS           DataLoader workers (default: 6)
#   MASTER_PORT_BASE      Base port (default: 29600)
#   DRY_RUN=1             Print commands without executing
#
set -euo pipefail

DIT_ROOT="${DIT_ROOT:-.}"
MANIFEST_BASE_DIR="${MANIFEST_BASE_DIR:-.}"
RSUT_JSON_TRAIN_DIR="${RSUT_JSON_TRAIN_DIR:-./data/Json/Train}"
RSUT_JSON_VALID_DIR="${RSUT_JSON_VALID_DIR:-./data/Json/Valid}"
RSUT_JSON_TEST_DIR="${RSUT_JSON_TEST_DIR:-./data/Json/Test}"

RSUT_RESULTS_BATCH="${RSUT_RESULTS_BATCH:-results}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50000}"
CKPT_EVERY="${CKPT_EVERY:-2500}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-384}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"

# VAE checkpoint paths (set these before running)
: "${AE_CKPT_SAR:=path/to/sar.ckpt}"
: "${AE_CKPT_OPT:=path/to/opt.ckpt}"
: "${AE_CKPT_MS:=path/to/ms.ckpt}"
: "${AE_CKPT_NIR:=path/to/nir.ckpt}"
: "${AE_CKPT_PAN:=path/to/pan.ckpt}"

if [[ -n "${SERVER_DIRECTIONS_FILE:-}" ]] && [[ -f "$SERVER_DIRECTIONS_FILE" ]]; then
  source "$SERVER_DIRECTIONS_FILE"
fi

_scale_for_mod() {
  case "$1" in
    sar) echo "0.422003" ;;
    opt) echo "0.387068" ;;
    ms) echo "0.484645" ;;
    nir) echo "0.568811" ;;
    pan) echo "0.447582" ;;
    *) echo "" ;;
  esac
}

_ae_for_mod() {
  case "$1" in
    opt) echo "$AE_CKPT_OPT" ;;
    sar) echo "$AE_CKPT_SAR" ;;
    pan) echo "$AE_CKPT_PAN" ;;
    ms) echo "$AE_CKPT_MS" ;;
    nir) echo "$AE_CKPT_NIR" ;;
    *) echo "" ;;
  esac
}

resolve_selected_direction_ids() {
  local raw="${SERVER_RUN_DIRECTIONS:-}"
  local -a tmp=()
  local p
  if [[ -z "${raw// }" ]]; then
    local i
    for ((i = 1; i <= 14; i++)); do tmp+=("$i"); done
  else
    IFS=',' read -ra parts <<< "$raw"
    for p in "${parts[@]}"; do
      p="${p// /}"
      [[ -z "$p" ]] && continue
      if ! [[ "$p" =~ ^[0-9]+$ ]]; then
        echo "Error: Invalid direction ID (must be 1-14): '$p'" >&2
        exit 1
      fi
      if ((p < 1 || p > 14)); then
        echo "Error: Direction ID must be 1-14: $p" >&2
        exit 1
      fi
      tmp+=("$p")
    done
  fi
  if ((${#tmp[@]} == 0)); then
    echo "Error: No valid direction IDs" >&2
    exit 1
  fi
  local line
  while IFS= read -r line; do
    [[ -n "$line" ]] && SELECTED_IDS+=("$line")
  done < <(printf '%s\n' "${tmp[@]}" | sort -n -u)
}

cd "$DIT_ROOT" || exit 1
[[ -f train.py ]] || { echo "Error: train.py not found. Set DIT_ROOT to project root." >&2; exit 1; }

LOG_ROOT="${LOG_ROOT:-${DIT_ROOT}/logs}"
mkdir -p "$LOG_ROOT"
SERIES_LOG="${LOG_ROOT}/series_${RSUT_RESULTS_BATCH}.log"

declare -a DIR_ROWS=(
  "Opt-Sar|opt|sar"
  "Opt-Sar|sar|opt"
  "Opt-Pan|opt|pan"
  "Opt-Pan|pan|opt"
  "Opt-Ms|opt|ms"
  "Opt-Ms|ms|opt"
  "Opt-Nir|opt|nir"
  "Opt-Nir|nir|opt"
  "Nir-MS|nir|ms"
  "Nir-MS|ms|nir"
  "Sar-Ms|sar|ms"
  "Sar-Ms|ms|sar"
  "Sar-Nir|sar|nir"
  "Sar-Nir|nir|sar"
)

declare -a SELECTED_IDS=()
resolve_selected_direction_ids

find_latest_checkpoint() {
  local results_sub="$1"
  local best_ckpt=""
  local best_step=-1
  local d c bn step

  shopt -s nullglob
  for d in "$results_sub"/*-DiT-S-4; do
    [[ -d "$d" ]] || continue
    for c in "$d"/checkpoints/*.pt; do
      [[ -f "$c" ]] || continue
      bn="${c##*/}"
      bn="${bn%.pt}"
      [[ "$bn" =~ ^[0-9]+$ ]] || continue
      step=$((10#$bn))
      if ((step > best_step)); then
        best_step="$step"
        best_ckpt="$c"
      fi
    done
  done
  shopt -u nullglob

  echo "$best_ckpt"
}

run_one() {
  local direction_id="$1"
  local row="$2"
  local pair_tag src tgt s1 s2 ae1 ae2 allowed run_tag results_sub train_json val_json test_json port logf resume_ckpt

  IFS='|' read -r pair_tag src tgt <<<"$row"
  s1="$(_scale_for_mod "$src")"
  s2="$(_scale_for_mod "$tgt")"
  ae1="$(_ae_for_mod "$src")"
  ae2="$(_ae_for_mod "$tgt")"
  if [[ -z "$s1" || -z "$s2" || -z "$ae1" || -z "$ae2" ]]; then
    echo "Error: Unknown modality pair_tag=$pair_tag src=$src tgt=$tgt" >&2
    exit 1
  fi

  allowed="${src}->${tgt}"
  run_tag="${pair_tag}__${src}_to_${tgt}"
  results_sub="${DIT_ROOT}/results/${RSUT_RESULTS_BATCH}/${run_tag}"
  train_json="${RSUT_JSON_TRAIN_DIR}/train_${pair_tag}.json"
  val_json="${RSUT_JSON_VALID_DIR}/valid_${pair_tag}.json"
  test_json="${RSUT_JSON_TEST_DIR}/test_${pair_tag}.json"

  for f in "$train_json" "$val_json" "$test_json" "$ae1" "$ae2"; do
    [[ -f "$f" ]] || { echo "Error: Missing file: $f" >&2; exit 1; }
  done

  mkdir -p "$results_sub"
  port=$((MASTER_PORT_BASE + direction_id))
  logf="${LOG_ROOT}/${run_tag}.log"
  resume_ckpt="$(find_latest_checkpoint "$results_sub")"

  {
    echo "==== $(date -Is) Direction #${direction_id} ${run_tag} MASTER_PORT=${port} ===="
    echo "max_train_steps=${MAX_TRAIN_STEPS} ckpt_every=${CKPT_EVERY} eval_every=${EVAL_EVERY}"
    if [[ -n "$resume_ckpt" ]]; then
      echo "auto_resume=ON resume_ckpt=${resume_ckpt}"
    else
      echo "auto_resume=ON resume_ckpt=<none>"
    fi
  } | tee -a "$SERIES_LOG" | tee -a "$logf"

  local -a cmd=(
    torchrun --nnodes=1 --nproc_per_node=8 --master_port="${port}" train.py
    --manifest-base-dir "$MANIFEST_BASE_DIR"
    --manifest-file "$train_json"
    --val-manifest-file "$val_json"
    --test-manifest-file "$test_json"
    --manifest-data-type raw
    --uni-transfer-root .
    --model DiT-S/4
    --latent-size 64
    --latent-channels 4
    --latent-scale-factors "${src}=${s1}" "${tgt}=${s2}"
    --conditioning-mode concat
    --modalities "$src" "$tgt"
    --ae-ckpts "${src}=${ae1}" "${tgt}=${ae2}"
    --require-ae-ckpts True
    --enable-any-to-all True
    --allowed-pairs "${allowed}"
    --no-resize True
    --augment True
    --cond-dropout-prob 0.0
    --modality-dropout-prob 0.0
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --predict-xstart True
    --x0-loss-snr-gamma 0
    --enable-after-adapter True
    --after-adapter-type refine
    --after-adapter-lr 1e-4
    --after-adapter-max-t 200
    --latent-loss-scale 1.0
    --pair-loss-reweight False
    --pair-loss-reweight-alpha 0.5
    --pair-loss-reweight-min 0.25
    --pair-loss-reweight-max 4.0
    --train-sampler balanced
    --balance-key pair
    --balanced-cover-all True
    --log-unused-samples
    --run-eval True
    --eval-preflight True
    --eval-preflight-before-opt True
    --eval-noadapt-baseline False
    --eval-enable-val False
    --eval-enable-test True
    --eval-enable-fid False
    --eval-enable-lpips False
    --eval-use-ema True
    --eval-clip-denoised False
    --eval-num-samples -1
    --eval-vis-samples 100
    --eval-batch-size 32
    --eval-eps-mse True
    --eval-eps-max-batch 16
    --eval-distributed True
    --lr 2.5e-5
    --eval-sampler ddim
    --eval-num-sampling-steps 250
    --eval-ddim-eta 0.0
    --monitor-plateau-window 5
    --monitor-x0-t999-min-improve 0.01
    --ckpt-every "$CKPT_EVERY"
    --eval-every "$EVAL_EVERY"
    --save-initial-checkpoint True
    --max-train-steps "$MAX_TRAIN_STEPS"
    --epochs 100000
    --results-dir "$results_sub"
  )

  if [[ -n "$resume_ckpt" ]]; then
    cmd+=(--resume "$resume_ckpt")
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]] || [[ "${DRY_RUN:-0}" == "true" ]]; then
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee -a "$logf"
  local ec=$?
  set -e
  if [[ "$ec" -ne 0 ]]; then
    echo "$(date -Is) Failed: Direction #${direction_id} ${run_tag} exit=${ec}" | tee -a "$SERIES_LOG" >&2
    exit "$ec"
  fi
  echo "$(date -Is) Completed: Direction #${direction_id} ${run_tag}" | tee -a "$SERIES_LOG"
}

{
  echo ""
  echo "======== $(date -Is) Task Plan ========"
  echo "RSUT_RESULTS_BATCH=${RSUT_RESULTS_BATCH}"
  echo "Directions: ${SELECTED_IDS[*]} (total ${#SELECTED_IDS[@]})"
  echo "Steps: ${MAX_TRAIN_STEPS}, GPUs: 8, ckpt every ${CKPT_EVERY}, eval every ${EVAL_EVERY}"
  echo "Port rule: MASTER_PORT = ${MASTER_PORT_BASE} + direction_id"
  for did in "${SELECTED_IDS[@]}"; do
    ai=$((did - 1))
    IFS='|' read -r pt s t <<<"${DIR_ROWS[$ai]}"
    echo "  #${did}  ${pt}  ${s}->${t}"
  done
  echo "======================================="
  echo ""
} | tee -a "$SERIES_LOG"

for direction_id in "${SELECTED_IDS[@]}"; do
  ai=$((direction_id - 1))
  run_one "$direction_id" "${DIR_ROWS[$ai]}"
done

echo "$(date -Is) All directions completed." | tee -a "$SERIES_LOG"
