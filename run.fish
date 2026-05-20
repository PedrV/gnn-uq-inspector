#!/usr/bin/fish

if test (count $argv) -lt 2
    echo "Usage: ./run.fish gpu|cpu dataset1 [dataset2 ...]"
    exit 1
end

set DEVICE $argv[1]
set DATASETS $argv[2..-1]

if not contains $DEVICE gpu cpu
    echo "Error: first argument must be 'gpu' or 'cpu'"
    exit 1
end


function run_job
    echo "Running: $argv"
    eval $argv
    sleep 2
end

for DS in $DATASETS
    switch $DS

        case pems
            run_job "./run_in_container.fish $DEVICE 1 train paths=container dataset=pems \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                task.validation_logic=loss \
                model.hidden_dim=100 \
                model.num_layers=3"

        case cora
            run_job "./run_in_container.fish $DEVICE 1 train paths=container dataset=cora \
                logging=no_forced \
                repetition.num_models=50 \
                repetition.num_repetitions=3"

        case citeseer
            run_job "./run_in_container.fish $DEVICE 1 train paths=container dataset=citeseer \
                logging=no_forced \
                repetition.num_models=50 \
                repetition.num_repetitions=3 \
                training.epochs=200"

        case gapsmallqm9
            run_job "./run_in_container.fish $DEVICE 1 train paths=container dataset=gapsmallqm9 \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=qm9_train \
                model=gcngraph"

        case artnetviews
            run_job "./run_in_container.fish $DEVICE 1 train paths=container dataset=artnetviews \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=graphland \
                model=megagcn"

            run_job "./run_in_container.fish $DEVICE 1 train paths=container dataset=artnetviews \
                logging=no_forced \
                dataset.version=shift_version \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=graphland \
                model=megagcn"


        case tolokers2
            run_job "./run_in_container.fish $DEVICE 1 train paths=container dataset=tolokers2 \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=graphland \
                model=megagat \
                model.hidden_dim=256"

            run_job "./run_in_container.fish $DEVICE 1 train paths=container dataset=tolokers2 \
                logging=no_forced \
                dataset.version=shift_version \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=graphland \
                model=megagat \
                model.hidden_dim=256"

        case chameleon
            run_job "./run_in_container.fish $DEVICE 1 train paths=container dataset=chameleon \
                logging=no_forced \
                repetition.num_models=10 \
                repetition.num_repetitions=5 \
                training=graphland \
                training.epochs=250 \
                model=megagat"

        case '*'
            echo "Warning: unknown dataset '$DS' — skipped"
    end
end
