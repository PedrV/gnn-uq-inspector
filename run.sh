#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: ./run.sh gpu|cpu dataset1 [dataset2 ...]"
    exit 1
fi

DEVICE="$1"
shift
DATASETS=("$@")

if [[ "$DEVICE" != "gpu" && "$DEVICE" != "cpu" ]]; then
    echo "Error: first argument must be 'gpu' or 'cpu'"
    exit 1
fi


run_job () {
    echo "Running: $*"
    eval "$@"
    sleep 2
}


for DS in "${DATASETS[@]}"; do
    case "$DS" in

        pems)
            run_job "./run_in_container.sh $DEVICE 1 train paths=container dataset=pems \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                task.validation_logic=loss \
                model.hidden_dim=100 \
                model.num_layers=3"
            ;;

        cora)
            run_job "./run_in_container.sh $DEVICE 1 train paths=container dataset=cora \
                logging=no_forced \
                repetition.num_models=50 \
                repetition.num_repetitions=3"
            ;;

        citeseer)
            run_job "./run_in_container.sh $DEVICE 1 train paths=container dataset=citeseer \
                logging=no_forced \
                repetition.num_models=50 \
                repetition.num_repetitions=3 \
                training.epochs=200"
            ;;

        gapsmallqm9)
            run_job "./run_in_container.sh $DEVICE 1 train paths=container dataset=gapsmallqm9 \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=qm9_train \
                model=gcngraph"
            ;;

        artnetviews)
            run_job "./run_in_container.sh $DEVICE 1 train paths=container dataset=artnetviews \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=graphland \
                model=megagcn"

            run_job "./run_in_container.sh $DEVICE 1 train paths=container dataset=artnetviews \
                dataset.version=shift_original \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=graphland \
                model=megagcn"
            ;;

        tolokers2)
            run_job "./run_in_container.sh $DEVICE 1 train paths=container dataset=tolokers2 \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=graphland \
                model=megagat \
                model.hidden_dim=256"

            run_job "./run_in_container.sh $DEVICE 1 train paths=container dataset=tolokers2 \
                dataset.version=shift_original \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=graphland \
                model=megagat \
                model.hidden_dim=256"
            ;;

        chameleon)
            run_job "./run_in_container.sh $DEVICE 1 train paths=container dataset=chameleon \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=graphland \
                training.epochs=250 \
                model=megagat"
            ;;

        *)
            echo "Warning: unknown dataset '$DS' — skipped"
            ;;
    esac
done
