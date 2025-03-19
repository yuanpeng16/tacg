# Theoretical Analysis for Systematic Generalization

## Main experiments

    sh experiments/xor/baseline.sh
    sh experiments/xor/proposed.sh

## Ablation experiments

    sh experiments/xor/no_regularization.sh
    sh experiments/xor/no_decoder.sh
    sh experiments/xor/lack_data.sh
    sh experiments/xor/architecture.sh

## Summarize results

    python3 summarize_results.py

## Analyze SCAN data

    python3 data_analysis.py