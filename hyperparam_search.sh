#!/bin/bash

# Array of PIDs of background processes
pids=()

# A function to kill all child processes on Ctrl+C
function cleanup {
    echo "Stopping all training processes..."
    for pid in "${pids[@]}"; do
        echo "Killing process $pid"
        kill -9 "$pid" 2>/dev/null
    done
    exit 1
}

trap cleanup SIGINT

declare -a policy_lr=(0.001 0.0005)
declare -a value_lr=(0.001 0.0005)
declare -a gamma=(0.95 0.99)

num_runs=${#policy_lr[@]}
run_num=0
for plr in "${policy_lr[@]}"
do
  for vlr in "${value_lr[@]}"
  do
    for g in "${gamma[@]}"
    do
      echo "Starting run $run_num with policy_lr=$plr, value_lr=$vlr, gamma=$g"
      python norwegian_chef_baseline_school.py \
        --policy_learning_rate "$plr" \
        --value_learning_rate "$vlr" \
        --gamma "$g" \
        --shape_factor 1 \
        --layout "cramped_room" \
        --seed 2 \
        --track True &
      pids+=($!)
      ((run_num++))
    done
  done
done
wait  # wait for all background jobs to finish
echo "All runs completed!"
