#!/bin/bash

MODELS=(
    "gpt-4o"
    "gpt-5.2"
    # "ayavision"
)

SETUPS=(
    "TextOnly"
    "ImageOnly"
    "ImageText"
)

for model in "${MODELS[@]}"; do
    for setup in "${SETUPS[@]}"; do
        echo "Running: model=$model | setup=$setup"
        # uv run python experiments/experiment_scoring.py \
        uv run python experiment_scoring.py \
            --model-name "$model" \
            --experiment-setup "$setup" \
            --no-llm-judge \
            --force
    done
done
