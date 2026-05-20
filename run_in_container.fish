#!/usr/bin/fish

set HW_TYPE $argv[1]
set CONTAINERID $argv[2]

# Check if $argv[3] is an action. If its NOT, we assume its a directory.
if not contains -- $argv[3] train do
    set EXTRA_DIR $argv[3]
    set ACTION    $argv[4]
    set PY_ARGS   $argv[5..-1]
else
    set EXTRA_DIR ""
    set ACTION    $argv[3]
    set PY_ARGS   $argv[4..-1]
end

set CHECK_BUILD true

if test -n "$EXTRA_DIR"
    set EXTRA_MOUNT -v "$EXTRA_DIR:/workspace/existing_run:z"
else
    set EXTRA_MOUNT
end

if test "$HW_TYPE" = "gpu"
    set IMAGE "localhost/gnndeeepensembles_gpu:latest"
    # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-podman
    # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
    # Do not worry about cuda version from nvidia-smi (that should list the one in the base system)
    # do python -c "import torch; print(torch.version.cuda)" to check the true version.
    # Generally base system version should be >= that the one in container.
    set GPU_FLAGS "--device" "nvidia.com/gpu=0" "--security-opt=label=disable"

    set ENV_VOLUME "venv_r_gpu:/workspace/venv_gpu_r"

    # https://docs.podman.io/en/v4.3/markdown/options/volume.html
    # If venv volume was modified e.g. installed a new package,
    # use podman volume rm venv before build or the very aggressive podman system prune --volumes (or --all)
    # If things keep breaking, rm -rf ~/.local/share/containers/storage/volumes/venv_gpu_r
    if test "$CHECK_BUILD" = "true"
        echo "--- Checking/Building GPU Image ---"
        podman build -t $IMAGE -f container/gpu-reduced/Containerfile .
    end

else if test "$HW_TYPE" = "cpu"
    set IMAGE "localhost/gnndeeepensembles_cpu:latest"
    set GPU_FLAGS

    set ENV_VOLUME "venv_r:/workspace/venv_r"

    # https://docs.podman.io/en/v4.3/markdown/options/volume.html
    # If venv volume was modified e.g. installed a new package,
    # use podman volume rm venv before build or the very aggressive podman system prune --volumes (or --all)
    # If things keep breaking, rm -rf ~/.local/share/containers/storage/volumes/venv_r
    if test "$CHECK_BUILD" = "true"
        echo "--- Checking/Building CPU Image ---"
        podman build -t $IMAGE -f container/cpu/Containerfile .
    end

else
    echo "Error: First argument must be 'cpu' or 'gpu'"
    exit 1
end


set BASE_COMMAND podman run --rm \
    --name trainer_pod_$CONTAINERID \
    --userns=keep-id \
    $GPU_FLAGS \
    -e PYTHONUNBUFFERED=1 \
    -v .:/workspace:z \
    -v ./outputs:/workspace/outputs:z \
    -v ./datasets:/workspace/datasets:z \
    $EXTRA_MOUNT \
    -v $ENV_VOLUME \
    $IMAGE

echo "--- Starting Task: $ACTION ---"
if test "$ACTION" = "train"
    $BASE_COMMAND python -m inspector.run $PY_ARGS
else if test "$ACTION" = "do"
    $BASE_COMMAND $PY_ARGS
else
    echo "Usage: ./run_in_container.fish [cpu|gpu] [container_id] [optional: extra_dir] [train|do] [args...]"
    exit 1
end
