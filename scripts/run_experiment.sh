#!/bin/bash
# ============================================================================
# run_experiment.sh — Stage-aware experiment launcher for NYC Metro
#
# Default pipeline:
#   [optional data prep] → LoRA training → inference 2021 + 2022 → evaluation
#
# Common usage:
#   bash run_experiment.sh --name exp02 --rebuild_data --prep_time 36:00:00
#   bash run_experiment.sh --name exp02 --rebuild_data --account torch_pr_196_tandon_advanced --partition h200_tandon --prep_account torch_pr_196_general --prep_partition cs
#   bash run_experiment.sh --name exp02 --stages train
#   bash run_experiment.sh --name exp02 --stages infer,eval
#   bash run_experiment.sh --name exp02 --stages infer --infer_years 2022
#   bash run_experiment.sh --name exp02 --stages prep --prep_skip_step_a --prep_skip_step_b
#   bash run_experiment.sh --name exp02 --stages train --resume_from_checkpoint auto
#   bash run_experiment.sh --name exp02_wcbg --rebuild_data --origin_mode work_cbg_centroid
#   bash run_experiment.sh --name exp02 --rebuild_data --mail_user sx2490@nyu.edu
#   # Hoist CPU stages (prep / inferprep / eval) INSIDE the GPU jobs so they
#   # bypass the shared CPU queue (general + cs / cpu_short). Adds CPU wall
#   # time to GPU jobs (GPU idle during CPU work) but eliminates queue wait:
#   bash run_experiment.sh --name exp02_wcbg --rebuild_data --origin_mode work_cbg_centroid --hoist_cpu_to_gpu
# ============================================================================

set -e
set -o pipefail
cd /scratch/sx2490/econai/nyc_metro
mkdir -p logs

# ── Defaults ─────────────────────────────────────────────────────────────────
NAME="exp02"
TRAIN_ROWS=300000
EPOCHS=1
BATCH=4
GRAD_ACCUM=8
LR="2e-4"
LORA_R=16
LORA_ALPHA=32
MAX_LEN=2048
MODEL="Qwen/Qwen2.5-7B-Instruct"
PARTITION="h200_tandon"
ACCOUNT="torch_pr_196_tandon_advanced"
PREP_ACCOUNT="torch_pr_196_general"
EVAL_ACCOUNT="torch_pr_196_general"
PREP_PARTITION="cs"
PREP_CPUS=4
PREP_MEM="16G"
EVAL_CPUS=1
EVAL_MEM="8G"
MAIL_USER="sx2490@nyu.edu"
MAIL_TYPE="END,FAIL,TIME_LIMIT"
INFER_K=30
TEMPERATURE=0.2
INFER_MAX_TOKENS=768
INFER_MAX_MODEL_LEN=4096
INFER_MIN_PARSE_RATE=0.95
INFER_PREP_RECORDS=1
INFER_PREP_CPUS=1
INFER_PREP_MEM="16G"
INFER_PREP_TIME="12:00:00"
INFER_MAX_SAMPLES_2021=50000
INFER_MAX_SAMPLES_2022=100000
EVAL_STEPS=2000
SAVE_STEPS=1000
MAX_EVAL=2000
VAL_DATE_FROM="2021-11-01"
ORIGIN_MODE="stop"
POLYGON_POLICY="all"
EXCLUDE_POI_CSV=""
EXCLUDE_TRUTH_POI_CSV=""
CANDIDATE_INJECT_POI_CSV=""
SYNTHETIC_UNSEEN_POI_CSV=""
POINT_MEMBER_AUTHORITY_CSV="data/poi_name_authority.csv"
REBUILD_DATA=0
STAGES="train,infer,eval"
INFER_YEARS="2021,2022"
BASELINE_METHODS="frequency,gravity,huff,mnl_grid"
BASELINE_TIME="08:00:00"
PREP_SKIP_STEP_A=0
PREP_SKIP_STEP_B=0
PRIMARY_RADIUS_M=1500
FALLBACK_RADIUS_M=3000
HARD_CAP_RADIUS_M=5000
PREP_TIME="24:00:00"
TRAIN_TIME="48:00:00"
INFER_TIME="24:00:00"
EVAL_TIME="04:00:00"
EVAL_PARTITION="cs"
RESUME_FROM_CHECKPOINT=""
# When set, CPU stages (prep / inferprep / eval) run *inside* the GPU SBATCH
# wraps instead of as separate CPU jobs. Avoids queueing on cs/cpu_short when
# the general account is blocked by other projects. GPU sits idle during the
# CPU portion of work, trading fairshare pressure for wall-clock latency.
HOIST_CPU_TO_GPU=0

validate_hpc_account() {
    local account="$1"
    local flag="$2"
    case "${account}" in
        torch_pr_196_general|torch_pr_196_tandon_advanced) ;;
        *)
            echo "ERROR: ${flag} must be one of: torch_pr_196_general, torch_pr_196_tandon_advanced"
            echo "       Got: ${account}"
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)              NAME="$2"; shift 2;;
        --train_rows)        TRAIN_ROWS="$2"; shift 2;;
        --epochs)            EPOCHS="$2"; shift 2;;
        --batch)             BATCH="$2"; shift 2;;
        --grad_accum)        GRAD_ACCUM="$2"; shift 2;;
        --lr)                LR="$2"; shift 2;;
        --lora_r)            LORA_R="$2"; shift 2;;
        --lora_alpha)        LORA_ALPHA="$2"; shift 2;;
        --max_len)           MAX_LEN="$2"; shift 2;;
        --model)             MODEL="$2"; shift 2;;
        --account)           ACCOUNT="$2"; shift 2;;
        --partition)         PARTITION="$2"; shift 2;;
        --prep_account)      PREP_ACCOUNT="$2"; shift 2;;
        --eval_account)      EVAL_ACCOUNT="$2"; shift 2;;
        --prep_partition)    PREP_PARTITION="$2"; shift 2;;
        --prep_cpus)         PREP_CPUS="$2"; shift 2;;
        --prep_mem)          PREP_MEM="$2"; shift 2;;
        --eval_cpus)         EVAL_CPUS="$2"; shift 2;;
        --eval_mem)          EVAL_MEM="$2"; shift 2;;
        --mail_user)         MAIL_USER="$2"; shift 2;;
        --mail_type)         MAIL_TYPE="$2"; shift 2;;
        --infer_k)           INFER_K="$2"; shift 2;;
        --temperature)       TEMPERATURE="$2"; shift 2;;
        --infer_max_tokens)  INFER_MAX_TOKENS="$2"; shift 2;;
        --infer_max_model_len) INFER_MAX_MODEL_LEN="$2"; shift 2;;
        --infer_min_parse_rate) INFER_MIN_PARSE_RATE="$2"; shift 2;;
        --no_infer_prep_records) INFER_PREP_RECORDS=0; shift 1;;
        --infer_prep_cpus)  INFER_PREP_CPUS="$2"; shift 2;;
        --infer_prep_mem)   INFER_PREP_MEM="$2"; shift 2;;
        --infer_prep_time)  INFER_PREP_TIME="$2"; shift 2;;
        --infer_max_samples_2021) INFER_MAX_SAMPLES_2021="$2"; shift 2;;
        --infer_max_samples_2022) INFER_MAX_SAMPLES_2022="$2"; shift 2;;
        --eval_steps)        EVAL_STEPS="$2"; shift 2;;
        --save_steps)        SAVE_STEPS="$2"; shift 2;;
        --max_eval)          MAX_EVAL="$2"; shift 2;;
        --val_date_from)     VAL_DATE_FROM="$2"; shift 2;;
        --origin_mode)       ORIGIN_MODE="$2"; shift 2;;
        --polygon_policy)    POLYGON_POLICY="$2"; shift 2;;
        --exclude_poi_csv)   EXCLUDE_POI_CSV="$2"; shift 2;;
        --exclude_truth_poi_csv) EXCLUDE_TRUTH_POI_CSV="$2"; shift 2;;
        --candidate_inject_poi_csv) CANDIDATE_INJECT_POI_CSV="$2"; shift 2;;
        --synthetic_unseen_poi_csv) SYNTHETIC_UNSEEN_POI_CSV="$2"; shift 2;;
        --point_member_authority_csv) POINT_MEMBER_AUTHORITY_CSV="$2"; shift 2;;
        --rebuild_data)      REBUILD_DATA=1; shift 1;;
        --stages)            STAGES="$2"; shift 2;;
        --infer_years)       INFER_YEARS="$2"; shift 2;;
        --baseline_methods)  BASELINE_METHODS="$2"; shift 2;;
        --baseline_time)     BASELINE_TIME="$2"; shift 2;;
        --prep_skip_step_a)  PREP_SKIP_STEP_A=1; shift 1;;
        --prep_skip_step_b)  PREP_SKIP_STEP_B=1; shift 1;;
        --primary_radius_m)  PRIMARY_RADIUS_M="$2"; shift 2;;
        --fallback_radius_m) FALLBACK_RADIUS_M="$2"; shift 2;;
        --hard_cap_radius_m) HARD_CAP_RADIUS_M="$2"; shift 2;;
        --prep_time)         PREP_TIME="$2"; shift 2;;
        --train_time)        TRAIN_TIME="$2"; shift 2;;
        --infer_time)        INFER_TIME="$2"; shift 2;;
        --eval_time)         EVAL_TIME="$2"; shift 2;;
        --eval_partition)    EVAL_PARTITION="$2"; shift 2;;
        --resume_from_checkpoint) RESUME_FROM_CHECKPOINT="$2"; shift 2;;
        --hoist_cpu_to_gpu)  HOIST_CPU_TO_GPU=1; shift 1;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

STAGES=$(echo "${STAGES}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')
INFER_YEARS=$(echo "${INFER_YEARS}" | tr -d ' ')
POLYGON_POLICY=$(echo "${POLYGON_POLICY}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')
BASELINE_METHODS=$(echo "${BASELINE_METHODS}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')

validate_hpc_account "${ACCOUNT}" "--account"
validate_hpc_account "${PREP_ACCOUNT}" "--prep_account"
validate_hpc_account "${EVAL_ACCOUNT}" "--eval_account"

case "${POLYGON_POLICY}" in
    all|exclude_shared_building|point_members) ;;
    *) echo "ERROR: --polygon_policy must be one of: all, exclude_shared_building, point_members"; exit 1;;
esac

stage_selected() {
    local needle="$1"
    [[ "${STAGES}" == "all" ]] || [[ ",${STAGES}," == *",${needle},"* ]]
}

year_selected() {
    local needle="$1"
    [[ ",${INFER_YEARS}," == *",${needle},"* ]]
}

WANT_PREP=0
WANT_TRAIN=0
WANT_INFER=0
WANT_EVAL=0
WANT_BASELINE=0

if [[ "${REBUILD_DATA}" == "1" ]] || stage_selected "prep"; then
    WANT_PREP=1
fi
if stage_selected "train"; then
    WANT_TRAIN=1
fi
if stage_selected "infer"; then
    WANT_INFER=1
fi
if stage_selected "eval"; then
    WANT_EVAL=1
fi
if stage_selected "baseline"; then
    WANT_BASELINE=1
fi

if [[ "${WANT_INFER}" == "1" ]] && ! year_selected "2021" && ! year_selected "2022"; then
    echo "ERROR: --infer_years must contain 2021 and/or 2022 when infer stage is selected."
    exit 1
fi
if [[ "${WANT_BASELINE}" == "1" ]] && ! year_selected "2021" && ! year_selected "2022"; then
    echo "ERROR: --infer_years must contain 2021 and/or 2022 when baseline stage is selected."
    exit 1
fi
if [[ "${WANT_BASELINE}" == "1" && "${WANT_INFER}" == "1" && "${INFER_PREP_RECORDS}" != "1" ]]; then
    echo "ERROR: baseline stage needs inference records; remove --no_infer_prep_records."
    exit 1
fi

# Hoist plan:
#   HOIST_PREP_INLINE      — prepend prep python call into train GPU wrap
#   HOIST_INFERPREP_INLINE — skip separate inferprep SBATCH; infer builds records itself
#   HOIST_EVAL_INLINE      — append eval python call into the last infer GPU wrap
HOIST_PREP_INLINE=0
HOIST_INFERPREP_INLINE=0
HOIST_EVAL_INLINE=0
HOIST_EVAL_TARGET_YEAR=""
if [[ "${HOIST_CPU_TO_GPU}" == "1" ]]; then
    if [[ "${WANT_PREP}" == "1" && "${WANT_TRAIN}" == "1" ]]; then
        HOIST_PREP_INLINE=1
        echo "WARNING: prep hoisted into train GPU wrap. NYU Torch admin runs a low-GPU-utilization"
        echo "         watchdog that auto-cancels jobs with <85% GPU usage over 2h rolling window."
        echo "         Prep takes 1-2h of pure CPU work, which WILL trigger the watchdog (see BUG-023"
        echo "         and 2026-04-15 kill of jobs 6273218/6273223). Recommended instead:"
        echo "           1. Run prep on login node manually (fastest), OR"
        echo "           2. Submit a separate prep SBATCH on cs partition (drop --hoist_cpu_to_gpu"
        echo "              or drop --rebuild_data + re-submit with --stages train,infer,eval)."
    elif [[ "${WANT_PREP}" == "1" ]]; then
        echo "WARNING: --hoist_cpu_to_gpu cannot hoist prep without train; prep will still use its own SBATCH."
    fi

    if [[ "${WANT_INFER}" == "1" && "${INFER_PREP_RECORDS}" == "1" ]]; then
        HOIST_INFERPREP_INLINE=1
    fi

    if [[ "${WANT_EVAL}" == "1" && "${WANT_INFER}" == "1" ]]; then
        if year_selected "2022"; then
            HOIST_EVAL_TARGET_YEAR="2022"
        else
            HOIST_EVAL_TARGET_YEAR="2021"
        fi
        HOIST_EVAL_INLINE=1
    elif [[ "${WANT_EVAL}" == "1" ]]; then
        echo "WARNING: --hoist_cpu_to_gpu cannot hoist eval without infer; eval will still use its own SBATCH."
    fi
fi

EXP_DIR="experiments/${NAME}"
LORA_DIR="${EXP_DIR}/lora"
PRED_TRAIN="${EXP_DIR}/pred_insample_2021.jsonl"
PRED_TEST="${EXP_DIR}/pred_oos_2022.jsonl"
EVAL_OUT_DIR="${EXP_DIR}/eval"
INPUT_DIR="${EXP_DIR}/inputs/train_prior"
CBG_POI_DIR="${INPUT_DIR}/cbg_poi"
PERSONAS_PATH="${INPUT_DIR}/personas_nyc.jsonl"
FALLBACK_POI_CSV="${INPUT_DIR}/poi_unique_food_with_freq.csv"
WORK_CBG_ORIGIN_CSV="${INPUT_DIR}/work_cbg_mean_origins.csv"
SNAPSHOT_MANIFEST="${INPUT_DIR}/manifest.json"
INFER_RECORD_DIR="${EXP_DIR}/infer_records"
mkdir -p "${EXP_DIR}" "${INPUT_DIR}" "${INFER_RECORD_DIR}"

VAL_ROWS=$((TRAIN_ROWS / 20))
if [[ "${VAL_ROWS}" -lt 1 ]]; then
    VAL_ROWS=1
fi

OVERLAY=/scratch/sx2490/pytorch-env/overlay-50G-10M.ext3
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

TRAIN_FILE="train.jsonl"
VAL_FILE="val.jsonl"
if [[ "${WANT_PREP}" == "1" ]]; then
    TRAIN_FILE="${EXP_DIR}/train.jsonl"
    VAL_FILE="${EXP_DIR}/val.jsonl"
elif [[ -f "${EXP_DIR}/train.jsonl" && -s "${EXP_DIR}/val.jsonl" ]]; then
    TRAIN_FILE="${EXP_DIR}/train.jsonl"
    VAL_FILE="${EXP_DIR}/val.jsonl"
fi

if [[ -f "${TRAIN_FILE}" ]]; then
    SOURCE_TRAIN="${TRAIN_FILE}"
    SOURCE_VAL="${VAL_FILE}"
    CURRENT_ROWS=$(wc -l < "${TRAIN_FILE}")
    if [[ "${TRAIN_ROWS}" -lt "${CURRENT_ROWS}" ]]; then
        TRAIN_FILE="${EXP_DIR}/train_${TRAIN_ROWS}.jsonl"
        VAL_FILE="${EXP_DIR}/val_${TRAIN_ROWS}.jsonl"
        if [[ ! -f "${TRAIN_FILE}" ]]; then
            head -${TRAIN_ROWS} "${SOURCE_TRAIN}" > "${TRAIN_FILE}"
            echo "Created ${TRAIN_FILE} (${TRAIN_ROWS} rows)"
        fi
        if [[ ! -f "${VAL_FILE}" ]]; then
            head -${VAL_ROWS} "${SOURCE_VAL}" > "${VAL_FILE}"
            echo "Created ${VAL_FILE} (${VAL_ROWS} rows)"
        fi
    fi
fi

require_nonempty_file() {
    local path="$1"
    if [[ ! -s "${path}" ]]; then
        echo "ERROR: required file is missing or empty: ${path}"
        exit 1
    fi
}

require_matching_file() {
    local pattern="$1"
    shopt -s nullglob
    local matches=(${pattern})
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "ERROR: required files matching pattern are missing: ${pattern}"
        exit 1
    fi
}

if [[ -n "${EXCLUDE_POI_CSV}" ]]; then
    require_nonempty_file "${EXCLUDE_POI_CSV}"
fi
if [[ -n "${EXCLUDE_TRUTH_POI_CSV}" ]]; then
    require_nonempty_file "${EXCLUDE_TRUTH_POI_CSV}"
fi
if [[ -n "${CANDIDATE_INJECT_POI_CSV}" ]]; then
    require_nonempty_file "${CANDIDATE_INJECT_POI_CSV}"
fi
if [[ -n "${SYNTHETIC_UNSEEN_POI_CSV}" ]]; then
    require_nonempty_file "${SYNTHETIC_UNSEEN_POI_CSV}"
fi
if [[ "${POLYGON_POLICY}" == "point_members" ]]; then
    require_nonempty_file "${POINT_MEMBER_AUTHORITY_CSV}"
fi

require_job_id() {
    local job_id="$1"
    local label="$2"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${label} sbatch did not return a valid job id: '${job_id}'"
        exit 1
    fi
}

if [[ "${WANT_PREP}" == "1" && "${PREP_SKIP_STEP_A}" == "1" ]]; then
    require_matching_file "${CBG_POI_DIR}/cluster_*_pois.csv"
    require_nonempty_file "${FALLBACK_POI_CSV}"
    if [[ "${ORIGIN_MODE}" == "work_cbg_centroid" ]]; then
        require_nonempty_file "${WORK_CBG_ORIGIN_CSV}"
    fi
fi

if [[ "${WANT_PREP}" == "1" && "${PREP_SKIP_STEP_B}" == "1" ]]; then
    require_nonempty_file "${PERSONAS_PATH}"
fi

if [[ "${WANT_TRAIN}" == "1" && "${WANT_PREP}" != "1" ]]; then
    require_nonempty_file "${TRAIN_FILE}"
    require_nonempty_file "${VAL_FILE}"
    require_matching_file "${CBG_POI_DIR}/cluster_*_pois.csv"
    require_nonempty_file "${PERSONAS_PATH}"
    require_nonempty_file "${FALLBACK_POI_CSV}"
    if [[ "${ORIGIN_MODE}" == "work_cbg_centroid" ]]; then
        require_nonempty_file "${WORK_CBG_ORIGIN_CSV}"
    fi
fi

if [[ "${WANT_INFER}" == "1" && "${WANT_TRAIN}" != "1" ]]; then
    require_nonempty_file "${LORA_DIR}/final/adapter_config.json"
    require_nonempty_file "${LORA_DIR}/final/adapter_model.safetensors"
    if [[ "${WANT_PREP}" != "1" || "${PREP_SKIP_STEP_A}" == "1" ]]; then
        require_matching_file "${CBG_POI_DIR}/cluster_*_pois.csv"
        require_nonempty_file "${FALLBACK_POI_CSV}"
        if [[ "${ORIGIN_MODE}" == "work_cbg_centroid" ]]; then
            require_nonempty_file "${WORK_CBG_ORIGIN_CSV}"
        fi
    fi
    if [[ "${WANT_PREP}" != "1" || "${PREP_SKIP_STEP_B}" == "1" ]]; then
        require_nonempty_file "${PERSONAS_PATH}"
    fi
fi

if [[ "${WANT_EVAL}" == "1" ]]; then
    if [[ "${WANT_PREP}" != "1" || "${PREP_SKIP_STEP_A}" == "1" ]]; then
        require_matching_file "${CBG_POI_DIR}/cluster_*_pois.csv"
        require_nonempty_file "${FALLBACK_POI_CSV}"
    fi
    if [[ "${WANT_INFER}" != "1" ]] || ! year_selected "2021"; then
        require_nonempty_file "${PRED_TRAIN}"
    fi
    if [[ "${WANT_INFER}" != "1" ]] || ! year_selected "2022"; then
        require_nonempty_file "${PRED_TEST}"
    fi
fi

if [[ "${WANT_BASELINE}" == "1" && "${WANT_PREP}" != "1" ]]; then
    require_matching_file "${CBG_POI_DIR}/cluster_*_pois.csv"
    require_nonempty_file "${PERSONAS_PATH}"
    require_nonempty_file "${FALLBACK_POI_CSV}"
    if [[ "${ORIGIN_MODE}" == "work_cbg_centroid" ]]; then
        require_nonempty_file "${WORK_CBG_ORIGIN_CSV}"
    fi
fi

if [[ "${WANT_PREP}" == "1" && "${WANT_TRAIN}" != "1" && ( "${WANT_INFER}" == "1" || "${WANT_EVAL}" == "1" ) ]]; then
    echo "WARNING: prep is selected without train; downstream infer/eval will keep using the existing LoRA weights."
fi

if [[ "${WANT_PREP}" == "1" && "${PREP_SKIP_STEP_B}" == "1" && "${PREP_SKIP_STEP_A}" != "1" ]]; then
    echo "WARNING: step A will rebuild cluster pools but step B is skipped; personas_nyc.jsonl may be stale relative to the new pools."
fi

EFF_BATCH=$((BATCH * GRAD_ACCUM))
STEPS=$((TRAIN_ROWS / EFF_BATCH * EPOCHS))

cat > "${EXP_DIR}/config.json" << EOF
{
  "name": "${NAME}",
  "stages": "${STAGES}",
  "infer_years": "${INFER_YEARS}",
  "prep_skip_step_a": ${PREP_SKIP_STEP_A},
  "prep_skip_step_b": ${PREP_SKIP_STEP_B},
  "resume_from_checkpoint": "${RESUME_FROM_CHECKPOINT}",
  "model": "${MODEL}",
  "train_rows": ${TRAIN_ROWS},
  "val_rows": ${VAL_ROWS},
  "epochs": ${EPOCHS},
  "batch_size": ${BATCH},
  "grad_accum": ${GRAD_ACCUM},
  "effective_batch": ${EFF_BATCH},
  "estimated_steps": ${STEPS},
  "learning_rate": "${LR}",
  "lora_r": ${LORA_R},
  "lora_alpha": ${LORA_ALPHA},
  "max_length": ${MAX_LEN},
  "eval_steps": ${EVAL_STEPS},
  "save_steps": ${SAVE_STEPS},
  "max_eval_samples": ${MAX_EVAL},
  "val_date_from": "${VAL_DATE_FROM}",
  "origin_mode": "${ORIGIN_MODE}",
  "polygon_policy": "${POLYGON_POLICY}",
  "exclude_poi_csv": "${EXCLUDE_POI_CSV}",
  "exclude_truth_poi_csv": "${EXCLUDE_TRUTH_POI_CSV}",
  "candidate_inject_poi_csv": "${CANDIDATE_INJECT_POI_CSV}",
  "synthetic_unseen_poi_csv": "${SYNTHETIC_UNSEEN_POI_CSV}",
  "point_member_authority_csv": "${POINT_MEMBER_AUTHORITY_CSV}",
  "rebuild_data": ${REBUILD_DATA},
  "primary_radius_m": ${PRIMARY_RADIUS_M},
  "fallback_radius_m": ${FALLBACK_RADIUS_M},
  "hard_cap_radius_m": ${HARD_CAP_RADIUS_M},
  "infer_k": ${INFER_K},
  "infer_max_samples_2021": ${INFER_MAX_SAMPLES_2021},
  "infer_max_samples_2022": ${INFER_MAX_SAMPLES_2022},
  "temperature": ${TEMPERATURE},
  "infer_max_tokens": ${INFER_MAX_TOKENS},
  "infer_max_model_len": ${INFER_MAX_MODEL_LEN},
  "infer_min_parse_rate": ${INFER_MIN_PARSE_RATE},
  "infer_prep_records": ${INFER_PREP_RECORDS},
  "infer_prep_cpus": ${INFER_PREP_CPUS},
  "infer_prep_mem": "${INFER_PREP_MEM}",
  "infer_prep_time": "${INFER_PREP_TIME}",
  "baseline_methods": "${BASELINE_METHODS}",
  "baseline_time": "${BASELINE_TIME}",
  "account": "${ACCOUNT}",
  "partition": "${PARTITION}",
  "prep_account": "${PREP_ACCOUNT}",
  "prep_partition": "${PREP_PARTITION}",
  "prep_cpus": ${PREP_CPUS},
  "prep_mem": "${PREP_MEM}",
  "eval_account": "${EVAL_ACCOUNT}",
  "eval_partition": "${EVAL_PARTITION}",
  "eval_cpus": ${EVAL_CPUS},
  "eval_mem": "${EVAL_MEM}",
  "mail_user": "${MAIL_USER}",
  "mail_type": "${MAIL_TYPE}",
  "prep_time": "${PREP_TIME}",
  "train_time": "${TRAIN_TIME}",
  "infer_time": "${INFER_TIME}",
  "eval_time": "${EVAL_TIME}",
  "snapshot_manifest": "${SNAPSHOT_MANIFEST}",
  "hoist_cpu_to_gpu": ${HOIST_CPU_TO_GPU},
  "hoist_prep_inline": ${HOIST_PREP_INLINE},
  "hoist_inferprep_inline": ${HOIST_INFERPREP_INLINE},
  "hoist_eval_inline": ${HOIST_EVAL_INLINE},
  "hoist_eval_target_year": "${HOIST_EVAL_TARGET_YEAR}"
}
EOF

echo "=== Experiment: ${NAME} ==="
echo "  Stages:         ${STAGES}"
echo "  Infer years:    ${INFER_YEARS}"
echo "  Rebuild data:   ${REBUILD_DATA}"
echo "  Prep skips:     step_a=${PREP_SKIP_STEP_A}, step_b=${PREP_SKIP_STEP_B}"
echo "  Model:          ${MODEL}"
echo "  Train/infer:    account=${ACCOUNT}, partition=${PARTITION}"
echo "  Prep:           account=${PREP_ACCOUNT}, partition=${PREP_PARTITION}, cpus=${PREP_CPUS}, mem=${PREP_MEM}, time=${PREP_TIME}"
echo "  Eval:           account=${EVAL_ACCOUNT}, partition=${EVAL_PARTITION}, cpus=${EVAL_CPUS}, mem=${EVAL_MEM}, time=${EVAL_TIME}"
echo "  Mail:           user=${MAIL_USER}, type=${MAIL_TYPE}"
echo "  Train:          ${TRAIN_ROWS} rows × ${EPOCHS} epochs"
echo "  Val rows:       ${VAL_ROWS}"
echo "  Origin mode:    ${ORIGIN_MODE}"
echo "  Polygon policy: ${POLYGON_POLICY}"
[[ -n "${EXCLUDE_POI_CSV}" ]] && echo "  Hidden train POIs: ${EXCLUDE_POI_CSV}"
[[ -n "${EXCLUDE_TRUTH_POI_CSV}" ]] && echo "  Exclude truth POIs: ${EXCLUDE_TRUTH_POI_CSV}"
[[ -n "${CANDIDATE_INJECT_POI_CSV}" ]] && echo "  Inject candidates: ${CANDIDATE_INJECT_POI_CSV}"
[[ -n "${SYNTHETIC_UNSEEN_POI_CSV}" ]] && echo "  Synthetic unseen eval: ${SYNTHETIC_UNSEEN_POI_CSV}"
echo "  Infer samples:  2021=${INFER_MAX_SAMPLES_2021}, 2022=${INFER_MAX_SAMPLES_2022}"
echo "  Infer parsing:  max_model_len=${INFER_MAX_MODEL_LEN}, max_tokens=${INFER_MAX_TOKENS}, min_parse_rate=${INFER_MIN_PARSE_RATE}"
echo "  Infer prep:     records=${INFER_PREP_RECORDS}, account=${PREP_ACCOUNT}, partition=${PREP_PARTITION}, cpus=${INFER_PREP_CPUS}, mem=${INFER_PREP_MEM}, time=${INFER_PREP_TIME}"
echo "  Hoist CPU→GPU:  enabled=${HOIST_CPU_TO_GPU}, prep_inline=${HOIST_PREP_INLINE}, inferprep_inline=${HOIST_INFERPREP_INLINE}, eval_inline=${HOIST_EVAL_INLINE}, eval_target_year=${HOIST_EVAL_TARGET_YEAR:-none}"
echo "  Batch:          ${BATCH} × ${GRAD_ACCUM} = ${EFF_BATCH} effective"
echo "  Steps:          ~${STEPS}"
echo "  LR:             ${LR}"
echo "  LoRA:           r=${LORA_R} α=${LORA_ALPHA}"
echo "  Eval during train: every ${EVAL_STEPS} steps, max ${MAX_EVAL} samples"
[[ "${WANT_BASELINE}" == "1" ]] && echo "  Baselines:      ${BASELINE_METHODS}"
echo "  Train file:     ${TRAIN_FILE}"
echo "  Val file:       ${VAL_FILE}"
echo "  Pred train:     ${PRED_TRAIN}"
echo "  Pred test:      ${PRED_TEST}"
echo "  Snapshot:       ${SNAPSHOT_MANIFEST}"
echo "  Output:         ${EXP_DIR}/"
echo ""

PREP_SKIP_A_ARG=""
if [[ "${PREP_SKIP_STEP_A}" == "1" ]]; then
    PREP_SKIP_A_ARG="--skip_step_a"
fi
PREP_SKIP_B_ARG=""
if [[ "${PREP_SKIP_STEP_B}" == "1" ]]; then
    PREP_SKIP_B_ARG="--skip_step_b"
fi
RESUME_ARG=""
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}"
fi
POLYGON_ARG="--polygon_policy ${POLYGON_POLICY} --point_member_authority_csv ${POINT_MEMBER_AUTHORITY_CSV}"
PREP_EXCLUDE_ARG=""
if [[ -n "${EXCLUDE_POI_CSV}" ]]; then
    PREP_EXCLUDE_ARG="--exclude_poi_csv ${EXCLUDE_POI_CSV}"
fi
INFER_EXCLUDE_TRUTH_ARG=""
if [[ -n "${EXCLUDE_TRUTH_POI_CSV}" ]]; then
    INFER_EXCLUDE_TRUTH_ARG="--exclude_truth_poi_csv ${EXCLUDE_TRUTH_POI_CSV}"
fi
INFER_INJECT_ARG=""
if [[ -n "${CANDIDATE_INJECT_POI_CSV}" ]]; then
    INFER_INJECT_ARG="--candidate_inject_poi_csv ${CANDIDATE_INJECT_POI_CSV}"
fi
EVAL_SYNTHETIC_ARG=""
if [[ -n "${SYNTHETIC_UNSEEN_POI_CSV}" ]]; then
    EVAL_SYNTHETIC_ARG="--synthetic_unseen_poi_csv ${SYNTHETIC_UNSEEN_POI_CSV}"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Hoist command blocks: these run INSIDE the GPU SBATCH wrap, OUTSIDE the
# singularity container, using the system conda (same path the CPU jobs use).
# The blocks are only injected when the corresponding HOIST_*_INLINE flag is 1.
# Newlines are preserved so the block drops cleanly into the surrounding wrap.
# ──────────────────────────────────────────────────────────────────────────────
PREP_HOIST_CMD=""
if [[ "${HOIST_PREP_INLINE}" == "1" ]]; then
    PREP_HOIST_CMD="
echo '>>> Hoisted prep (running on GPU node, outside singularity) <<<'
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate base
python3 -u 10_sft_data_prep.py \\
    --a_work data/a_work_2021.csv \\
    --cbg_poi_dir ${CBG_POI_DIR} \\
    --personas ${PERSONAS_PATH} \\
    --fallback_poi_csv ${FALLBACK_POI_CSV} \\
    --work_cbg_origin_csv ${WORK_CBG_ORIGIN_CSV} \\
    --snapshot_manifest ${SNAPSHOT_MANIFEST} \\
    --out_train ${EXP_DIR}/train.jsonl \\
    --out_val ${EXP_DIR}/val.jsonl \\
    --k 30 \\
    --max_train_rows ${TRAIN_ROWS} \\
    --max_val_rows ${VAL_ROWS} \\
    --val_date_from ${VAL_DATE_FROM} \\
    --train_prior_end_date ${VAL_DATE_FROM} \\
    --origin_mode ${ORIGIN_MODE} \\
    ${POLYGON_ARG} \\
    ${PREP_EXCLUDE_ARG} \\
    --primary_radius_m ${PRIMARY_RADIUS_M} \\
    --fallback_radius_m ${FALLBACK_RADIUS_M} \\
    --hard_cap_radius_m ${HARD_CAP_RADIUS_M} \\
    ${PREP_SKIP_A_ARG} \\
    ${PREP_SKIP_B_ARG}
echo '>>> Hoisted prep done <<<'
"
fi

EVAL_HOIST_CMD=""
if [[ "${HOIST_EVAL_INLINE}" == "1" ]]; then
    EVAL_HOIST_CMD="
echo '>>> Hoisted eval (running on GPU node, outside singularity) <<<'
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate base
export MPLCONFIGDIR=/tmp/mpl_${NAME}_hoist_\${SLURM_JOB_ID}
mkdir -p \${MPLCONFIGDIR}
python3 -u 14_temporal_eval.py \\
    --exp ${NAME} \\
    --pred_train ${PRED_TRAIN} \\
    --pred_test ${PRED_TEST} \\
    --snapshot_manifest ${SNAPSHOT_MANIFEST} \\
    ${EVAL_SYNTHETIC_ARG} \\
    --out_dir ${EVAL_OUT_DIR}
echo '>>> Hoisted eval done <<<'
"
fi

JOB_PREP=""
if [[ "${WANT_PREP}" == "1" && "${HOIST_PREP_INLINE}" == "1" ]]; then
    echo "=== Stage: prep (hoisted into train GPU job, no separate SBATCH) ==="
elif [[ "${WANT_PREP}" == "1" ]]; then
    echo "=== Stage: prep ==="
    JOB_PREP=$(sbatch \
        --job-name="prep_${NAME}" \
        --account=${PREP_ACCOUNT} --partition=${PREP_PARTITION} \
        --cpus-per-task=${PREP_CPUS} --mem=${PREP_MEM} --time=${PREP_TIME} \
        --mail-user=${MAIL_USER} --mail-type=${MAIL_TYPE} \
        --output="logs/prep_${NAME}_%j.log" --error="logs/prep_${NAME}_%j.err" \
        --wrap="
set -e
cd /scratch/sx2490/econai/nyc_metro
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate base

echo \"=== Data Prep ${NAME}: \$(date) ===\"
	python3 -u 10_sft_data_prep.py \
	    --a_work data/a_work_2021.csv \
	    --cbg_poi_dir ${CBG_POI_DIR} \
	    --personas ${PERSONAS_PATH} \
	    --fallback_poi_csv ${FALLBACK_POI_CSV} \
	    --work_cbg_origin_csv ${WORK_CBG_ORIGIN_CSV} \
	    --snapshot_manifest ${SNAPSHOT_MANIFEST} \
	    --out_train ${EXP_DIR}/train.jsonl \
	    --out_val ${EXP_DIR}/val.jsonl \
	    --k 30 \
	    --max_train_rows ${TRAIN_ROWS} \
	    --max_val_rows ${VAL_ROWS} \
	    --val_date_from ${VAL_DATE_FROM} \
	    --train_prior_end_date ${VAL_DATE_FROM} \
	    --origin_mode ${ORIGIN_MODE} \
	    ${POLYGON_ARG} \
	    ${PREP_EXCLUDE_ARG} \
	    --primary_radius_m ${PRIMARY_RADIUS_M} \
	    --fallback_radius_m ${FALLBACK_RADIUS_M} \
	    --hard_cap_radius_m ${HARD_CAP_RADIUS_M} \
	    --n_workers ${PREP_CPUS} \
    ${PREP_SKIP_A_ARG} \
    ${PREP_SKIP_B_ARG}
echo \"=== Data Prep Done: \$(date) ===\"
" | awk '{print $4}')
    require_job_id "${JOB_PREP}" "prep"
    echo "  Prep job: ${JOB_PREP}"
fi

JOB_TRAIN=""
if [[ "${WANT_TRAIN}" == "1" ]]; then
    echo "=== Stage: train ==="
    TRAIN_DEP=""
    if [[ -n "${JOB_PREP}" ]]; then
        TRAIN_DEP="--dependency=afterok:${JOB_PREP}"
    fi
    JOB_TRAIN=$(sbatch ${TRAIN_DEP} \
        --job-name="train_${NAME}" \
        --account=${ACCOUNT} --partition=${PARTITION} \
        --cpus-per-task=16 --mem=128G --gres=gpu:1 --time=${TRAIN_TIME} \
        --mail-user=${MAIL_USER} --mail-type=${MAIL_TYPE} \
        --output="logs/train_${NAME}_%j.log" --error="logs/train_${NAME}_%j.err" \
        --wrap="
set -e
cd /scratch/sx2490/econai/nyc_metro
${PREP_HOIST_CMD}
singularity exec --nv --overlay ${OVERLAY}:ro ${SIF} /bin/bash -c '
source /ext3/env.sh 2>/dev/null || true
export PYTHONNOUSERSITE=1  # prevent ~/.local from shadowing overlay torch (migration fix)
export HF_HOME=/scratch/sx2490/hf_cache
export TRANSFORMERS_CACHE=/scratch/sx2490/hf_cache
cd /scratch/sx2490/econai/nyc_metro

set -e
echo \"=== Training ${NAME}: \$(date) ===\"
echo \"GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)\"

python3 -u 11_train_lora.py \
    --model_path ${MODEL} \
    --train_jsonl ${TRAIN_FILE} \
    --val_jsonl ${VAL_FILE} \
    --output_dir ${LORA_DIR} \
    --max_length ${MAX_LEN} \
    --per_device_train_batch_size ${BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_eval_samples ${MAX_EVAL} \
    --gradient_checkpointing \
    ${RESUME_ARG}

echo \"=== Training Done: \$(date) ===\"
'
" | awk '{print $4}')
    require_job_id "${JOB_TRAIN}" "train"
    echo "  Train job: ${JOB_TRAIN}"
fi

JOB_INFER_2021=""
JOB_INFER_2022=""
JOB_INFERPREP_2021=""
JOB_INFERPREP_2022=""
if [[ "${WANT_INFER}" == "1" ]]; then
    echo "=== Stage: infer ==="

    for YEAR in 2021 2022; do
        if ! year_selected "${YEAR}"; then
            continue
        fi
        LABEL=$( [[ "${YEAR}" == "2021" ]] && echo "insample" || echo "oos" )
        YEAR_MAX_SAMPLES=${INFER_MAX_SAMPLES_2022}
        if [[ "${YEAR}" == "2021" ]]; then
            YEAR_MAX_SAMPLES=${INFER_MAX_SAMPLES_2021}
        fi
        RECORDS_PATH="${INFER_RECORD_DIR}/records_${LABEL}_${YEAR}.jsonl"
        RECORDS_IN_ARG=""
        JOB_RECORD_PREP=""

        if [[ "${INFER_PREP_RECORDS}" == "1" && "${HOIST_INFERPREP_INLINE}" == "1" ]]; then
            # Hoisted: let infer itself build records on the fly; no separate
            # CPU job, no --records_in reuse. infer re-reads a_work CSVs and
            # samples visits inside the GPU job.
            RECORDS_IN_ARG=""
            echo "  Infer prep ${YEAR}: hoisted into GPU job"
        elif [[ "${INFER_PREP_RECORDS}" == "1" ]]; then
            RECORD_PREP_DEP=""
            if [[ -n "${JOB_PREP}" ]]; then
                RECORD_PREP_DEP="--dependency=afterok:${JOB_PREP}"
            fi
            JOB_RECORD_PREP=$(sbatch ${RECORD_PREP_DEP} \
                --job-name="inferprep_${NAME}_${YEAR}" \
                --account=${PREP_ACCOUNT} --partition=${PREP_PARTITION} \
                --cpus-per-task=${INFER_PREP_CPUS} --mem=${INFER_PREP_MEM} --time=${INFER_PREP_TIME} \
                --mail-user=${MAIL_USER} --mail-type=${MAIL_TYPE} \
                --output="logs/inferprep_${NAME}_${YEAR}_%j.log" \
                --error="logs/inferprep_${NAME}_${YEAR}_%j.err" \
                --wrap="
singularity exec --overlay ${OVERLAY}:ro ${SIF} /bin/bash -c '
source /ext3/env.sh 2>/dev/null || true
export PYTHONNOUSERSITE=1  # prevent ~/.local from shadowing overlay torch (migration fix)
export HF_HOME=/scratch/sx2490/hf_cache
export TRANSFORMERS_CACHE=/scratch/sx2490/hf_cache
cd /scratch/sx2490/econai/nyc_metro

set -e
echo \"=== Inference Prep ${NAME} ${YEAR}: \$(date) ===\"

python3 -u 12_infer_vllm.py \
    --year ${YEAR} \
    --model_path ${MODEL} \
    --lora_path ${LORA_DIR}/final \
    --out ${EXP_DIR}/pred_${LABEL}_${YEAR}.jsonl \
    --cbg_poi_dir ${CBG_POI_DIR} \
    --personas ${PERSONAS_PATH} \
    --fallback_poi_csv ${FALLBACK_POI_CSV} \
    --work_cbg_origin_csv ${WORK_CBG_ORIGIN_CSV} \
    --records_out ${RECORDS_PATH} \
    --prepare_only \
    --origin_mode ${ORIGIN_MODE} \
    ${POLYGON_ARG} \
    ${INFER_EXCLUDE_TRUTH_ARG} \
    ${INFER_INJECT_ARG} \
    --sampling_unit visit \
    --val_date_from ${VAL_DATE_FROM} \
    --max_samples ${YEAR_MAX_SAMPLES} \
    --k ${INFER_K} \
    --primary_radius_m ${PRIMARY_RADIUS_M} \
    --fallback_radius_m ${FALLBACK_RADIUS_M} \
    --hard_cap_radius_m ${HARD_CAP_RADIUS_M} \
    --n_workers ${INFER_PREP_CPUS}

echo \"=== Inference Prep Done ${NAME} ${YEAR}: \$(date) ===\"
'
" | awk '{print $4}')
            require_job_id "${JOB_RECORD_PREP}" "inferprep_${YEAR}"
            RECORDS_IN_ARG="--records_in ${RECORDS_PATH}"
            if [[ "${YEAR}" == "2021" ]]; then
                JOB_INFERPREP_2021="${JOB_RECORD_PREP}"
            else
                JOB_INFERPREP_2022="${JOB_RECORD_PREP}"
            fi
            echo "  Infer prep ${YEAR} job: ${JOB_RECORD_PREP}"
        fi

        INFER_DEP_IDS=()
        if [[ -n "${JOB_TRAIN}" ]]; then
            INFER_DEP_IDS+=("${JOB_TRAIN}")
        fi
        if [[ -n "${JOB_RECORD_PREP}" ]]; then
            INFER_DEP_IDS+=("${JOB_RECORD_PREP}")
        fi
        INFER_DEP=""
        if [[ ${#INFER_DEP_IDS[@]} -gt 0 ]]; then
            INFER_DEP_JOINED=$(IFS=:; echo "${INFER_DEP_IDS[*]}")
            INFER_DEP="--dependency=afterok:${INFER_DEP_JOINED}"
        fi
        YEAR_EVAL_TAIL=""
        if [[ "${HOIST_EVAL_INLINE}" == "1" && "${YEAR}" == "${HOIST_EVAL_TARGET_YEAR}" ]]; then
            YEAR_EVAL_TAIL="${EVAL_HOIST_CMD}"
        fi
        JOB_ID=$(sbatch ${INFER_DEP} \
            --job-name="infer_${NAME}_${YEAR}" \
            --account=${ACCOUNT} --partition=${PARTITION} \
            --cpus-per-task=16 --mem=128G --gres=gpu:1 --time=${INFER_TIME} \
            --mail-user=${MAIL_USER} --mail-type=${MAIL_TYPE} \
            --output="logs/infer_${NAME}_${YEAR}_%j.log" \
            --error="logs/infer_${NAME}_${YEAR}_%j.err" \
            --wrap="
set -e
cd /scratch/sx2490/econai/nyc_metro
singularity exec --nv --overlay ${OVERLAY}:ro ${SIF} /bin/bash -c '
source /ext3/env.sh 2>/dev/null || true
export PYTHONNOUSERSITE=1  # prevent ~/.local from shadowing overlay torch (migration fix)
export HF_HOME=/scratch/sx2490/hf_cache
export TRANSFORMERS_CACHE=/scratch/sx2490/hf_cache
cd /scratch/sx2490/econai/nyc_metro

set -e
echo \"=== Inference ${NAME} ${YEAR}: \$(date) ===\"

python3 -u 12_infer_vllm.py \
    --year ${YEAR} \
    --model_path ${MODEL} \
    --lora_path ${LORA_DIR}/final \
    --out ${EXP_DIR}/pred_${LABEL}_${YEAR}.jsonl \
    --cbg_poi_dir ${CBG_POI_DIR} \
    --personas ${PERSONAS_PATH} \
    --fallback_poi_csv ${FALLBACK_POI_CSV} \
    --work_cbg_origin_csv ${WORK_CBG_ORIGIN_CSV} \
    ${RECORDS_IN_ARG} \
    --origin_mode ${ORIGIN_MODE} \
    ${POLYGON_ARG} \
    ${INFER_EXCLUDE_TRUTH_ARG} \
    ${INFER_INJECT_ARG} \
    --sampling_unit visit \
    --val_date_from ${VAL_DATE_FROM} \
    --max_samples ${YEAR_MAX_SAMPLES} \
    --k ${INFER_K} \
    --temperature ${TEMPERATURE} \
    --max_tokens ${INFER_MAX_TOKENS} \
    --max_model_len ${INFER_MAX_MODEL_LEN} \
    --min_parse_rate ${INFER_MIN_PARSE_RATE} \
    --chunk_size 256 \
    --gpu_mem 0.90 \
    --primary_radius_m ${PRIMARY_RADIUS_M} \
    --fallback_radius_m ${FALLBACK_RADIUS_M} \
    --hard_cap_radius_m ${HARD_CAP_RADIUS_M}

echo \"=== Inference Done ${NAME} ${YEAR}: \$(date) ===\"
'
${YEAR_EVAL_TAIL}
" | awk '{print $4}')
        require_job_id "${JOB_ID}" "infer_${YEAR}"
        if [[ "${YEAR}" == "2021" ]]; then
            JOB_INFER_2021="${JOB_ID}"
        else
            JOB_INFER_2022="${JOB_ID}"
        fi
        echo "  Infer ${YEAR} job: ${JOB_ID}"
    done
fi

JOB_BASELINE=""
if [[ "${WANT_BASELINE}" == "1" ]]; then
    echo "=== Stage: baseline ==="
    BASELINE_DEP_IDS=()
    if [[ -n "${JOB_PREP}" ]]; then
        BASELINE_DEP_IDS+=("${JOB_PREP}")
    fi
    if year_selected "2021" && [[ -n "${JOB_INFERPREP_2021}" ]]; then
        BASELINE_DEP_IDS+=("${JOB_INFERPREP_2021}")
    fi
    if year_selected "2022" && [[ -n "${JOB_INFERPREP_2022}" ]]; then
        BASELINE_DEP_IDS+=("${JOB_INFERPREP_2022}")
    fi
    BASELINE_DEP=""
    if [[ ${#BASELINE_DEP_IDS[@]} -gt 0 ]]; then
        BASELINE_DEP_JOINED=$(IFS=:; echo "${BASELINE_DEP_IDS[*]}")
        BASELINE_DEP="--dependency=afterok:${BASELINE_DEP_JOINED}"
    fi
    BASELINE_DO_2021=0
    BASELINE_DO_2022=0
    if year_selected "2021"; then
        BASELINE_DO_2021=1
    fi
    if year_selected "2022"; then
        BASELINE_DO_2022=1
    fi

    JOB_BASELINE=$(sbatch ${BASELINE_DEP} \
        --job-name="base_${NAME}" \
        --account=${EVAL_ACCOUNT} --partition=${EVAL_PARTITION} \
        --cpus-per-task=${EVAL_CPUS} --mem=${EVAL_MEM} --time=${BASELINE_TIME} \
        --mail-user=${MAIL_USER} --mail-type=${MAIL_TYPE} \
        --output="logs/baseline_${NAME}_%j.log" --error="logs/baseline_${NAME}_%j.err" \
        --wrap="
set -e
cd /scratch/sx2490/econai/nyc_metro
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate base

export MPLCONFIGDIR=/tmp/mpl_${NAME}_baseline_\${SLURM_JOB_ID}
mkdir -p \${MPLCONFIGDIR}

echo \"=== Baselines ${NAME}: \$(date) ===\"

if [[ ${BASELINE_DO_2021} -eq 1 ]]; then
    if [[ ! -s ${INFER_RECORD_DIR}/records_insample_2021.jsonl ]] || ! grep -m 1 -q '\"candidates\"' ${INFER_RECORD_DIR}/records_insample_2021.jsonl; then
        echo \"Preparing baseline records for 2021...\"
        python3 -u 12_infer_vllm.py \
            --year 2021 \
            --model_path ${MODEL} \
            --lora_path ${LORA_DIR}/final \
            --out ${EXP_DIR}/pred_insample_2021.jsonl \
            --cbg_poi_dir ${CBG_POI_DIR} \
            --personas ${PERSONAS_PATH} \
            --fallback_poi_csv ${FALLBACK_POI_CSV} \
            --work_cbg_origin_csv ${WORK_CBG_ORIGIN_CSV} \
            --records_out ${INFER_RECORD_DIR}/records_insample_2021.jsonl \
            --prepare_only \
            --origin_mode ${ORIGIN_MODE} \
            ${POLYGON_ARG} \
            ${INFER_EXCLUDE_TRUTH_ARG} \
            ${INFER_INJECT_ARG} \
            --sampling_unit visit \
            --val_date_from ${VAL_DATE_FROM} \
            --max_samples ${INFER_MAX_SAMPLES_2021} \
            --k ${INFER_K} \
            --primary_radius_m ${PRIMARY_RADIUS_M} \
            --fallback_radius_m ${FALLBACK_RADIUS_M} \
            --hard_cap_radius_m ${HARD_CAP_RADIUS_M}
    fi
fi

if [[ ${BASELINE_DO_2022} -eq 1 ]]; then
    if [[ ! -s ${INFER_RECORD_DIR}/records_oos_2022.jsonl ]] || ! grep -m 1 -q '\"candidates\"' ${INFER_RECORD_DIR}/records_oos_2022.jsonl; then
        echo \"Preparing baseline records for 2022...\"
        python3 -u 12_infer_vllm.py \
            --year 2022 \
            --model_path ${MODEL} \
            --lora_path ${LORA_DIR}/final \
            --out ${EXP_DIR}/pred_oos_2022.jsonl \
            --cbg_poi_dir ${CBG_POI_DIR} \
            --personas ${PERSONAS_PATH} \
            --fallback_poi_csv ${FALLBACK_POI_CSV} \
            --work_cbg_origin_csv ${WORK_CBG_ORIGIN_CSV} \
            --records_out ${INFER_RECORD_DIR}/records_oos_2022.jsonl \
            --prepare_only \
            --origin_mode ${ORIGIN_MODE} \
            ${POLYGON_ARG} \
            ${INFER_EXCLUDE_TRUTH_ARG} \
            ${INFER_INJECT_ARG} \
            --sampling_unit visit \
            --val_date_from ${VAL_DATE_FROM} \
            --max_samples ${INFER_MAX_SAMPLES_2022} \
            --k ${INFER_K} \
            --primary_radius_m ${PRIMARY_RADIUS_M} \
            --fallback_radius_m ${FALLBACK_RADIUS_M} \
            --hard_cap_radius_m ${HARD_CAP_RADIUS_M}
    fi
fi

IFS=',' read -r -a METHOD_ARRAY <<< \"${BASELINE_METHODS}\"
FIT_ARG=\"\"
if [[ -s ${INFER_RECORD_DIR}/records_insample_2021.jsonl ]]; then
    FIT_ARG=\"--fit_records_in ${INFER_RECORD_DIR}/records_insample_2021.jsonl\"
fi
# Pin grid ranges here so baseline fits do not silently revert to a smaller
# default when 17_classical_baselines.py is updated elsewhere (see BUG-028).
GRID_ALPHAS=\"0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0\"
GRID_BETAS=\"0.0,0.01,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0\"
for METHOD in \"\${METHOD_ARRAY[@]}\"; do
    METHOD_DIR=\"${EXP_DIR}/baselines/\${METHOD}\"
    mkdir -p \"\${METHOD_DIR}\"
    if [[ ${BASELINE_DO_2021} -eq 1 ]]; then
        python3 -u 17_classical_baselines.py \
            --method \"\${METHOD}\" \
            --records_in ${INFER_RECORD_DIR}/records_insample_2021.jsonl \
            \${FIT_ARG} \
            --grid_alphas \"\${GRID_ALPHAS}\" --grid_betas \"\${GRID_BETAS}\" \
            --out \"\${METHOD_DIR}/pred_insample_2021.jsonl\"
    fi
    if [[ ${BASELINE_DO_2022} -eq 1 ]]; then
        python3 -u 17_classical_baselines.py \
            --method \"\${METHOD}\" \
            --records_in ${INFER_RECORD_DIR}/records_oos_2022.jsonl \
            \${FIT_ARG} \
            --grid_alphas \"\${GRID_ALPHAS}\" --grid_betas \"\${GRID_BETAS}\" \
            --out \"\${METHOD_DIR}/pred_oos_2022.jsonl\"
    fi
    if [[ -s \"\${METHOD_DIR}/pred_insample_2021.jsonl\" && -s \"\${METHOD_DIR}/pred_oos_2022.jsonl\" ]]; then
        python3 -u 14_temporal_eval.py \
            --exp \"${NAME}_baseline_\${METHOD}\" \
            --pred_train \"\${METHOD_DIR}/pred_insample_2021.jsonl\" \
            --pred_test \"\${METHOD_DIR}/pred_oos_2022.jsonl\" \
            --snapshot_manifest ${SNAPSHOT_MANIFEST} \
            ${EVAL_SYNTHETIC_ARG} \
            --out_dir \"\${METHOD_DIR}/eval\"
    fi
done
echo \"=== Baselines Done ${NAME}: \$(date) ===\"
" | awk '{print $4}')
    require_job_id "${JOB_BASELINE}" "baseline"
    echo "  Baseline job: ${JOB_BASELINE}"
fi

JOB_EVAL=""
if [[ "${WANT_EVAL}" == "1" && "${HOIST_EVAL_INLINE}" == "1" ]]; then
    echo "=== Stage: eval (hoisted into infer_${HOIST_EVAL_TARGET_YEAR} GPU job, no separate SBATCH) ==="
elif [[ "${WANT_EVAL}" == "1" ]]; then
    echo "=== Stage: eval ==="
    EVAL_DEP=""
    EVAL_DEP_IDS=()
    if [[ -n "${JOB_INFER_2021}" ]]; then
        EVAL_DEP_IDS+=("${JOB_INFER_2021}")
    fi
    if [[ -n "${JOB_INFER_2022}" ]]; then
        EVAL_DEP_IDS+=("${JOB_INFER_2022}")
    fi
    if [[ ${#EVAL_DEP_IDS[@]} -gt 0 ]]; then
        DEP_JOINED=$(IFS=:; echo "${EVAL_DEP_IDS[*]}")
        EVAL_DEP="--dependency=afterok:${DEP_JOINED}"
    fi

    JOB_EVAL=$(sbatch ${EVAL_DEP} \
        --job-name="eval_${NAME}" \
        --account=${EVAL_ACCOUNT} --partition=${EVAL_PARTITION} \
        --cpus-per-task=${EVAL_CPUS} --mem=${EVAL_MEM} --time=${EVAL_TIME} \
        --mail-user=${MAIL_USER} --mail-type=${MAIL_TYPE} \
        --output="logs/eval_${NAME}_%j.log" --error="logs/eval_${NAME}_%j.err" \
        --wrap="
set -e
cd /scratch/sx2490/econai/nyc_metro
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate base

export MPLCONFIGDIR=/tmp/mpl_${NAME}_\${SLURM_JOB_ID}
mkdir -p \${MPLCONFIGDIR}

echo \"=== Eval ${NAME}: \$(date) ===\"
python3 -u 14_temporal_eval.py \
    --exp ${NAME} \
    --pred_train ${PRED_TRAIN} \
    --pred_test ${PRED_TEST} \
    --snapshot_manifest ${SNAPSHOT_MANIFEST} \
    ${EVAL_SYNTHETIC_ARG} \
    --out_dir ${EVAL_OUT_DIR}
echo \"=== Eval Done ${NAME}: \$(date) ===\"
" | awk '{print $4}')
    require_job_id "${JOB_EVAL}" "eval"
    echo "  Eval job: ${JOB_EVAL}"
fi

echo ""
echo "Monitor: squeue -u sx2490"
echo "Config:  cat ${EXP_DIR}/config.json"
echo "Jobs:"
if [[ "${HOIST_PREP_INLINE}" == "1" ]]; then
    echo "  prep  (hoisted into train_${NAME})"
elif [[ -n "${JOB_PREP}" ]]; then
    echo "  prep  ${JOB_PREP}"
fi
[[ -n "${JOB_TRAIN}" ]] && echo "  train ${JOB_TRAIN}"
if [[ "${HOIST_INFERPREP_INLINE}" == "1" && "${WANT_INFER}" == "1" ]]; then
    echo "  inferprep (hoisted into infer_${NAME}_*)"
else
    [[ -n "${JOB_INFERPREP_2021}" ]] && echo "  inferprep 2021 ${JOB_INFERPREP_2021}"
    [[ -n "${JOB_INFERPREP_2022}" ]] && echo "  inferprep 2022 ${JOB_INFERPREP_2022}"
fi
[[ -n "${JOB_INFER_2021}" ]] && echo "  infer 2021 ${JOB_INFER_2021}"
[[ -n "${JOB_INFER_2022}" ]] && echo "  infer 2022 ${JOB_INFER_2022}"
[[ -n "${JOB_BASELINE}" ]] && echo "  baseline ${JOB_BASELINE}"
if [[ "${HOIST_EVAL_INLINE}" == "1" ]]; then
    echo "  eval  (hoisted into infer_${NAME}_${HOIST_EVAL_TARGET_YEAR})"
elif [[ -n "${JOB_EVAL}" ]]; then
    echo "  eval  ${JOB_EVAL}"
fi
