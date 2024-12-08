#!/bin/bash

# Array of possible values for N
N_values=(3 5 8)

# Base command template
base_command="python community.py --num_members 90 --num_turns 100 --group_task_distribution 7 --task_distribution_difficulty hard --group_abilities_distribution 7 --abilities_distribution_difficulty hard"

# Array of abilities to exclude
groups=("g1" "g2" "g3" "g4" "g5" "g6" "g7" "g8" "g9" "g10")

# Loop through values of N
for N in "${N_values[@]}"; do
    # Loop through each group to exclude
    for exclude in "${groups[@]}"; do
        # Build the groups portion of the command, excluding the current group
        groups_command=""
        for group in "${groups[@]}"; do
            if [[ "$group" != "$exclude" ]]; then
                groups_command+=" --$group 10"
            fi
        done

        # Construct the full command
        command="$base_command --num_abilities $N $groups_command"

        # Construct the log filename
        log_file="logs/exclude_${exclude}_n${N}_7hard.log"

        # Run the command with nohup
        nohup $command > $log_file 2>&1 &

        # Output a message to indicate the command is running
        echo "Started: Excluding $exclude, logging to $log_file"
    done
done

echo "All processes started."

