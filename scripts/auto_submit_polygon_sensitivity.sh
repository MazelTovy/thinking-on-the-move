#!/usr/bin/env bash
# Poll for Step C prep completion, then auto-submit train+infer+eval (no --rebuild_data
# to avoid BUG-026 GPU watchdog on hoisted prep).

set -u
cd /scratch/sx2490/econai/nyc_metro

STAMP=$(date +%Y%m%d_%H%M%S)
WLOG="logs/auto_submit_polygon_sensitivity_${STAMP}.log"
mkdir -p logs
exec >"$WLOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

# (exp_name, polygon_policy) pairs
EXPS=(
  "exp02_no_shared:exclude_shared_building"
  "exp02_point_members:point_members"
)

declare -A DONE
for e in "${EXPS[@]}"; do DONE[${e%%:*}]=0; done

log "Watching for prep completion. Log: $WLOG"

count_lines() {
  local f="$1"
  [[ -s "$f" ]] || { echo 0; return; }
  wc -l < "$f"
}

is_ready() {
  local exp="$1"
  local train="experiments/${exp}/train.jsonl"
  local val="experiments/${exp}/val.jsonl"
  [[ -s "$val" ]] || return 1
  local n nv
  n=$(count_lines "$train")
  nv=$(count_lines "$val")
  [[ "$n" -ge 300000 ]] || return 1
  [[ "$nv" -ge 15000 ]] || return 1
  return 0
}

prep_proc_alive() {
  local exp="$1"
  pgrep -af "10_sft_data_prep.*experiments/${exp}/" >/dev/null
}

submit_stages() {
  local exp="$1" policy="$2"
  log "Submitting train+infer+eval for $exp (policy=$policy)"
  bash run_experiment.sh \
    --name "$exp" \
    --stages train,infer,eval \
    --origin_mode work_cbg_centroid \
    --polygon_policy "$policy" \
    --hoist_cpu_to_gpu 2>&1
  local rc=$?
  log "run_experiment.sh exit=$rc for $exp"
  return $rc
}

while true; do
  all_done=1
  for pair in "${EXPS[@]}"; do
    exp="${pair%%:*}"
    policy="${pair##*:}"
    [[ "${DONE[$exp]}" == "1" ]] && continue
    all_done=0
    if is_ready "$exp"; then
      log "$exp: prep complete, submitting"
      if submit_stages "$exp" "$policy"; then
        DONE[$exp]=1
      else
        log "$exp: submission failed, will retry in 5 min"
      fi
    else
      train="experiments/${exp}/train.jsonl"
      val="experiments/${exp}/val.jsonl"
      nt=$(count_lines "$train")
      nv=$(count_lines "$val")
      alive="no"
      prep_proc_alive "$exp" && alive="yes"
      log "$exp: train=$nt val=$nv alive=$alive"
      if [[ "$alive" == "no" && "$nt" -lt 300000 ]]; then
        log "WARN: $exp prep process missing AND train incomplete ($nt/300k). Flag for manual restart."
      fi
    fi
  done
  [[ "$all_done" == "1" ]] && { log "All experiments submitted. Exiting watcher."; break; }
  sleep 300
done
