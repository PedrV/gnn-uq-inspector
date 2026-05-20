#!/bin/bash

HW_TYPE=$1
CONTAINERID=$2

if [[ "$3" != "train" && "$3" != "do" ]]; then
    EXTRA_DIR=$3
    ACTION=$4
    PY_ARGS=("${@:5}")
else
    EXTRA_DIR=""
    ACTION=$3
    PY_ARGS=("${@:4}")
fi

CHECK_BUILD=false

if [ -n "$EXTRA_DIR" ]; then
    EXTRA_MOUNT=(-v "$EXTRA_DIR:/workspace/existing_run:z")
else
    EXTRA_MOUNT=()
fi

if [ "$HW_TYPE" = "gpu" ]; then
    IMAGE="localhost/gnndeeepensembles_gpu:latest"
    # Using an array for flags to handle multiple space-separated arguments correctly
    GPU_FLAGS=("--device" "nvidia.com/gpu=0" "--security-opt=label=disable")
    ENV_VOLUME="venv_gpu:/workspace/venv_gpu_r"

    if [ "$CHECK_BUILD" = "true" ]; then
        echo "--- Checking/Building GPU Image ---"
        podman build -t "$IMAGE" -f container/gpu-reduced/Containerfile .
    fi

elif [ "$HW_TYPE" = "cpu" ]; then
    IMAGE="localhost/gnndeeepensembles_cpu:latest"
    GPU_FLAGS=()
    ENV_VOLUME="venv:/workspace/venv_r"

    if [ "$CHECK_BUILD" = "true" ]; then
        echo "--- Checking/Building CPU Image ---"
        podman build -t "$IMAGE" -f container/cpu/Containerfile .
    fi

else
    echo "Error: First argument must be 'cpu' or 'gpu'"
    exit 1
fi

BASE_COMMAND=(
    podman run --rm
    --name "trainer_pod_$CONTAINERID"
    --userns=keep-id
    "${GPU_FLAGS[@]}"
    -e PYTHONUNBUFFERED=1
    -v ".:/workspace:z"
    -v "./outputs:/workspace/outputs:z"
    -v "./datasets:/workspace/datasets:z"
    "${EXTRA_MOUNT[@]}"
    -v "$ENV_VOLUME"
    "$IMAGE"
)

echo "--- Starting Task: $ACTION ---"

if [ "$ACTION" = "train" ]; then
    "${BASE_COMMAND[@]}" python -m inspector.run "${PY_ARGS[@]}"
elif [ "$ACTION" = "do" ]; then
    "${BASE_COMMAND[@]}" "${PY_ARGS[@]}"
else
    echo "Usage: $0 [cpu|gpu] [container_id] [optional: extra_dir] [train|do] [args...]"
    exit 1
fi
