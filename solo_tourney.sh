#!/bin/bash

# Array of possible values for A
values=(1 2 3 5 6 7 8 9 10)

# Array of possible values for N
N_values=(3 5 8)

# Base command template
base_command="python community.py --num_members 45 --num_turns 100 --group_task_distribution 10 --task_distribution_difficulty hard --group_abilities_distribution 10 --abilities_distribution_difficulty hard"

# Loop through all values of N and A
for N in "${N_values[@]}"; do
    for A in "${values[@]}"; do
        # Construct the specific command
        command="$base_command --num_abilities $N --g$A 45"
                
        # Construct the log filename
        log_file="minitourney/10hard/g${A}_n${N}.log"

        # Run the command with nohup
        nohup $command > $log_file 2>&1 &
                
        # Output a message to indicate the command is running
        echo "Started: $command, logging to $log_file"
    done
done

echo "All processes started."

