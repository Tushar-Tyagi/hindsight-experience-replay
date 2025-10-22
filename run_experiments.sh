#!/bin/bash

# Hindsight Experience Replay (HER) Experiment Runner
# This script runs experiments on all Fetch environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if MPI is available
if ! command -v mpirun &> /dev/null; then
    print_warning "MPI not found. Running single-process experiments."
    USE_MPI=false
else
    print_status "MPI found. Running multi-process experiments."
    USE_MPI=true
fi

# Create necessary directories
mkdir -p saved_models experiments figures

# Function to run experiment
run_experiment() {
    local env_name=$1
    local n_processes=$2
    local n_epochs=$3
    local log_file="experiments/${env_name}_$(date +%Y%m%d_%H%M%S).log"
    
    print_status "Running experiment: $env_name with $n_processes processes for $n_epochs epochs"
    
    if [ "$USE_MPI" = true ] && [ "$n_processes" -gt 1 ]; then
        mpirun -np $n_processes python train.py \
            --env-name=$env_name \
            --n-epochs=$n_epochs \
            --tensorboard \
            2>&1 | tee $log_file
    else
        python train.py \
            --env-name=$env_name \
            --n-epochs=$n_epochs \
            --tensorboard \
            2>&1 | tee $log_file
    fi
    
    if [ $? -eq 0 ]; then
        print_status "Experiment $env_name completed successfully"
    else
        print_error "Experiment $env_name failed"
        return 1
    fi
}

# Function to run demo
run_demo() {
    local env_name=$1
    local epoch=$2
    
    print_status "Running demo for $env_name (epoch $epoch)"
    
    python demo.py \
        --env-name=$env_name \
        --epoch=$epoch \
        --n-episodes=5 \
        --render
    
    if [ $? -eq 0 ]; then
        print_status "Demo $env_name completed successfully"
    else
        print_error "Demo $env_name failed"
        return 1
    fi
}

# Main execution
main() {
    print_status "Starting HER experiments..."
    
    # Test the implementation first
    print_status "Testing implementation..."
    python test_her.py
    
    if [ $? -ne 0 ]; then
        print_error "Implementation test failed. Please fix issues before running experiments."
        exit 1
    fi
    
    print_status "Implementation test passed. Starting experiments...\n"
    
    # Run experiments for different environments
    experiments=(
        "FetchReach-v1:1:20"
        "FetchPush-v1:8:50"
        "FetchPickAndPlace-v1:16:50"
        "FetchSlide-v1:8:100"
    )
    
    for exp in "${experiments[@]}"; do
        IFS=':' read -r env_name n_processes n_epochs <<< "$exp"
        
        print_status "Starting experiment: $env_name"
        run_experiment $env_name $n_processes $n_epochs
        
        if [ $? -eq 0 ]; then
            print_status "Running demo for $env_name..."
            run_demo $env_name $n_epochs
        fi
        
        echo "----------------------------------------"
    done
    
    print_status "All experiments completed!"
    print_status "Check the 'experiments/' directory for logs and 'saved_models/' for trained models."
    print_status "Use 'tensorboard --logdir=runs' to view training progress."
}

# Handle command line arguments
case "${1:-all}" in
    "test")
        python test_her.py
        ;;
    "reach")
        run_experiment "FetchReach-v1" 1 20
        run_demo "FetchReach-v1" 20
        ;;
    "push")
        run_experiment "FetchPush-v1" 8 50
        run_demo "FetchPush-v1" 50
        ;;
    "pick")
        run_experiment "FetchPickAndPlace-v1" 16 50
        run_demo "FetchPickAndPlace-v1" 50
        ;;
    "slide")
        run_experiment "FetchSlide-v1" 8 100
        run_demo "FetchSlide-v1" 100
        ;;
    "demo")
        if [ -z "$2" ]; then
            print_error "Please specify environment name for demo"
            echo "Usage: $0 demo <env_name> [epoch]"
            exit 1
        fi
        run_demo "$2" "${3:-50}"
        ;;
    "all")
        main
        ;;
    *)
        echo "Usage: $0 [test|reach|push|pick|slide|demo|all]"
        echo ""
        echo "Commands:"
        echo "  test    - Run implementation tests"
        echo "  reach   - Run FetchReach-v1 experiment"
        echo "  push    - Run FetchPush-v1 experiment"
        echo "  pick    - Run FetchPickAndPlace-v1 experiment"
        echo "  slide   - Run FetchSlide-v1 experiment"
        echo "  demo    - Run demo for specified environment"
        echo "  all     - Run all experiments (default)"
        echo ""
        echo "Examples:"
        echo "  $0 test"
        echo "  $0 reach"
        echo "  $0 demo FetchReach-v1 20"
        exit 1
        ;;
esac
