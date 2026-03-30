#!/bin/bash
# Batch run all ablation experiments

echo "=========================================="
echo "Batch ablation experiments: lambda_p and lambda_t sensitivity analysis"
echo "=========================================="

cd ./src

# Define experiment configurations (lambda_p, lambda_t)
# Format: "lambda_p lambda_t"
EXPERIMENTS=(
    # Baseline
    "0.2 0.2"
    
    # lambda_p sensitivity analysis (lambda_t=0.2 fixed)
    "0.0 0.2"
    "0.1 0.2"
    "0.3 0.2"
    "0.4 0.2"
    
    # lambda_t sensitivity analysis (lambda_p=0.2 fixed)
    "0.2 0.0"
    "0.2 0.1"
    "0.2 0.3"
    "0.2 0.4"
    
    # Joint variation
    "0.1 0.1"
    "0.3 0.3"
    "0.1 0.3"
    "0.3 0.1"
)

echo "Total ${#EXPERIMENTS[@]} experiments"
echo ""

# Run each experiment
for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    read -r lp lt <<< "$exp"
    
    echo "=========================================="
    echo "[$((i+1))/${#EXPERIMENTS[@]}] Running experiment: lp=$lp, lt=$lt"
    echo "=========================================="
    
    bash run_scripts/ablation_reward_weights.sh $lp $lt
    
    if [ $? -ne 0 ]; then
        echo "Experiment lp=$lp, lt=$lt failed!"
        # Optional: continue to next experiment or stop
        # exit 1  # uncomment to stop on failure
    fi
    
    echo ""
    echo "Waiting 30 seconds before next experiment..."
    sleep 30
done

echo ""
echo "=========================================="
echo "All ablation experiments complete!"
echo "=========================================="
