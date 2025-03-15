#!/bin/bash

# Function to start a tmux session and run a command
start_tmux_session() {
    session_name=$1
    command=$2

    # Check if the session already exists
    if ! tmux has-session -t $session_name 2>/dev/null; then
        echo "Starting tmux session: $session_name"
        tmux new-session -d -s $session_name
        sleep 2  # Give tmux time to start the session
    fi

    # Send the command to the tmux session
    tmux send-keys -t $session_name "$command" C-m
}

# 1. Publish Map
start_tmux_session "servers" "roslaunch heartmet_demo_item_delivery item_delivery_servers.launch"

# 2. Start the map session and launch the map
start_tmux_session "map" "roslaunch mas_environments map.launch"

# 3. Start the joypad session and run commands
start_tmux_session "joypad" "rosnode kill /hsrb/interactive_teleop"
start_tmux_session "joypad" "rosnode kill /hsrb/interactive_teleop_joy"
start_tmux_session "joypad" "roslaunch mas_hsr_teleop joy.launch"

# 4. Start the navigation session and run commands
start_tmux_session "navigation" "rosnode kill /pose_integrator"
start_tmux_session "navigation" "roslaunch hsrb_rosnav_config hsrb_nav.launch"

echo "All tasks have been started in separate tmux sessions."

# Optionally, attach to a session (e.g., the map session)
tmux attach-session -t servers
