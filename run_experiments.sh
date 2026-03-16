#!/bin/bash
# ReCode experiment launcher — runs all tracks + safety baselines
#
# Usage:
#   ./run_experiments.sh          # launch all 4
#   ./run_experiments.sh track-a  # just Track A
#   ./run_experiments.sh track-b  # just Track B
#   ./run_experiments.sh safety   # just safety baselines

set -e
source .venv/bin/activate

SLICES="S1 S2 S3 S4 S5 S6 S7 S8"

run_track_a() {
    echo "=== Track A: GDPval only (no safety signal) ==="
    python -m src.evolve.run_evolve \
        --seed openai --tier slow \
        --slices $SLICES \
        --iterations 5 --sample-size 3 \
        --verbose
}

run_track_b() {
    echo "=== Track B: 50% GDPval + 50% ToolEmu safety ==="
    python -m src.evolve.run_evolve \
        --seed openai --tier slow \
        --slices $SLICES \
        --safety-weight 0.5 \
        --iterations 5 --sample-size 3 \
        --verbose
}

run_safety_baselines() {
    echo "=== Safety baseline: OpenAI seed (gpt-5.4) ==="
    python scripts/run_safety_eval.py \
        src/evolve/seeds/openai.py \
        --model gpt-5.4 --verbose \
        --output results/safety_baseline_openai

    echo "=== Safety baseline: Anthropic seed (claude-opus-4-6) ==="
    python scripts/run_safety_eval.py \
        src/evolve/seeds/anthropic.py \
        --model claude-opus-4-6 --verbose \
        --output results/safety_baseline_anthropic
}

case "${1:-all}" in
    track-a)  run_track_a ;;
    track-b)  run_track_b ;;
    safety)   run_safety_baselines ;;
    all)
        echo "Launch in separate terminals:"
        echo "  Terminal 1: ./run_experiments.sh track-a"
        echo "  Terminal 2: ./run_experiments.sh track-b"
        echo "  Terminal 3: ./run_experiments.sh safety"
        ;;
    *)
        echo "Usage: ./run_experiments.sh [track-a|track-b|safety|all]"
        exit 1
        ;;
esac
