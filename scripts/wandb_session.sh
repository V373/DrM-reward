#!/usr/bin/env bash

usage() {
    cat <<'EOF'
Usage:
  source scripts/wandb_session.sh

Prompts for a W&B API key without echoing it, then overrides the W&B account in
the current terminal. It does not modify ~/.netrc or the global W&B login.
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Error: this script must be sourced to modify the current terminal." >&2
    usage >&2
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    return 0
fi

if [[ ! -t 0 ]]; then
    echo "Error: run this script from an interactive terminal." >&2
    return 1
fi

read -r -s -p "Temporary W&B API key: " WANDB_API_KEY
printf '\n'

if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "Error: API key cannot be empty." >&2
    unset WANDB_API_KEY
    return 1
fi

export WANDB_API_KEY
export WANDB_ENTITY="${WANDB_ENTITY:-Vik373}"
export WANDB_PROJECT="${WANDB_PROJECT:-DrM-Yao}"

echo "Current terminal now uses W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "Run 'unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT' to restore ~/.netrc credentials."