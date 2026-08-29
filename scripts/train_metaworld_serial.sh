#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

failed_tasks=()

wait_for_process_group() {
    local process_group="$1"
    local shell_group

    # setsid should give every task its own process group.  Avoid waiting on
    # the shell's own group if process-group discovery races with task exit.
    shell_group="$(ps -o pgid= -p "$$" 2>/dev/null || true)"
    shell_group="${shell_group//[[:space:]]/}"
    if [[ -z "${process_group}" || "${process_group}" == "${shell_group}" ]]; then
        return
    fi

    while kill -0 -- "-${process_group}" 2>/dev/null; do
        echo "Waiting for remaining processes in process group ${process_group} ..."
        sleep 1
    done
}

run_task() {
    local task_name="$1"
    local run_pid process_group run_status

    # Put the launcher and all of its descendants in an isolated process
    # group, so a failed Python process cannot overlap with the next task.
    setsid conda run --no-capture-output -n drm-cu128 \
        python train_mw.py "task=${task_name}" agent=drm_mw &
    run_pid=$!

    process_group="$(ps -o pgid= -p "${run_pid}" 2>/dev/null || true)"
    process_group="${process_group//[[:space:]]/}"

    if wait "${run_pid}"; then
        run_status=0
    else
        run_status=$?
    fi

    # wait may return when the launcher exits while a worker is still
    # shutting down; do not start the next task until the whole group is gone.
    wait_for_process_group "${process_group}"
    return "${run_status}"
}

for task in assembly stick-pull disassemble coffee-push sweep-into \
            button-press-wall soccer-meta window-close drawer-open \
            door-lock hammer-meta; do
    echo "========== Starting MetaWorld task: ${task} =========="
    if run_task "${task}"; then
        echo "========== Finished MetaWorld task: ${task} =========="
    else
        status=$?
        failed_tasks+=("${task}(exit=${status})")
        echo "========== MetaWorld task ${task} failed with exit=${status}; continuing =========="
    fi
done

if ((${#failed_tasks[@]} > 0)); then
    echo "Completed with failed tasks: ${failed_tasks[*]}"
    exit 1
fi

echo "All MetaWorld tasks finished successfully."
