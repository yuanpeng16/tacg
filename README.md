# A Theoretical Analysis of Compositional Generalization in Neural Networks: A Necessary and Sufficient Condition

## Main experiments

    sh experiments/xor/baseline.sh
    sh experiments/xor/proposed.sh

## Ablation experiments

    sh experiments/xor/no_regularization.sh
    sh experiments/xor/no_structure.sh
    sh experiments/xor/lack_training_data.sh

## Summarize results

    python3 summarize_results.py

## SCAN Analysis

    # Download data
    git clone https://github.com/brendenlake/SCAN.git
    
    # Analyze
    python3 sentence_syntax_analysis.py