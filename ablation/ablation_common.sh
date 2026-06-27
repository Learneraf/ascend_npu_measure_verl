#!/usr/bin/env bash

ABLATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-$(cd "${ABLATION_DIR}/.." && pwd)}"
TABLES="${TABLES:-${BASE}/results_tables.md}"
OUTPUTS="${OUTPUTS:-${BASE}/outputs}"

ablation_maybe_activate_venv() {
    if [ "${USE_VENV:-0}" != "1" ]; then
        return 0
    fi

    if [ -f "${BASE}/venv/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "${BASE}/venv/bin/activate"
    else
        echo "[warn] USE_VENV=1 but ${BASE}/venv/bin/activate was not found; using current Python."
    fi
}

ablation_latest_log() {
    local pattern="$1"
    local files=()
    local latest=""

    shopt -s nullglob
    files=("${OUTPUTS}"/${pattern})
    shopt -u nullglob

    if [ ${#files[@]} -eq 0 ]; then
        return 0
    fi

    for file in "${files[@]}"; do
        if [ -z "${latest}" ] || [ "${file}" -nt "${latest}" ]; then
            latest="${file}"
        fi
    done

    printf '%s\n' "${latest}"
}

ablation_fill_latest() {
    local pattern="$1"
    local mode="$2"
    local label="${3:-${mode}}"
    local latest_log

    latest_log="$(ablation_latest_log "${pattern}" || true)"
    if [ -z "${latest_log}" ]; then
        echo "[warn] No log matched ${OUTPUTS}/${pattern}; skipping fill for ${label}"
        return 1
    fi

    ablation_maybe_activate_venv
    (cd "${BASE}" && python3 fill_tables.py --log "${latest_log}" --tables "${TABLES}" --mode "${mode}")
    echo "[fill] ${label} updated from ${latest_log}"
}

ablation_mark_oom() {
    local mode="$1"
    local label="${2:-${mode}}"

    ablation_maybe_activate_venv
    (cd "${BASE}" && python3 fill_tables.py --tables "${TABLES}" --mode "${mode}")
    echo "[fill] ${label} marked as failed/OOM"
}

ablation_run_32b() {
    actor_param_offload=${actor_param_offload:-False} \
    actor_optimizer_offload=${actor_optimizer_offload:-True} \
    rollout_enforce_eager=${rollout_enforce_eager:-True} \
    rollout_enable_prefix_caching=${rollout_enable_prefix_caching:-False} \
    rollout_enable_chunked_prefill=${rollout_enable_chunked_prefill:-False} \
    gpu_memory_utilization=${gpu_memory_utilization:-0.85} \
    bash "${BASE}/run_qwen3_32b_lora_a100.sh" "$@"
}

ablation_cleanup() {
    echo "[cleanup] Killing stale VLLM::* processes..."
    pkill -9 -f 'VLLM::' 2>/dev/null || true
    sleep "${ABLATION_CLEANUP_SLEEP:-3}"
    echo "[cleanup] Done."
}
