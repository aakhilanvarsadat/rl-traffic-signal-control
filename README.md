# rl-traffic-signal-control
Reinforcement learning–based traffic signal control using DQN, Double DQN, and Dueling DQN on a realistic Dublin junction with SCATS-style baseline comparison. Includes training, evaluation, KPI analysis, and trade-off visualisations to study how algorithm choice and training length affect delay, throughput, and smoothness.

Final Model Inference — RL Sweep Comparison

This module runs the trained RL traffic-signal controllers (DQN, Double DQN, Dueling DQN) in SUMO and compares their performance on test scenarios.

1) Prerequisites

Python 3.8+

SUMO installed (sumo and/or sumo-gui in PATH)

Install Python dependencies:

pip install numpy pandas matplotlib torch traci

2) Running the comparison

From inside the Final_MI directory:

chmod +x run_rl_sweep_compare.sh   # make the script executable (only once)
./run_rl_sweep_compare.sh


The script will:

Load the correct network and test routes

Run each RL controller at the specified training checkpoints (200, 400, 1000 episodes)

Collect KPIs such as average waiting time, throughput, and approach speed

Save results in the configured output folder (default: compare_out/)

3) Output

Simulation logs and metrics CSVs are stored in compare_out/

Plots comparing controllers are generated in the same folder

The terminal will display a quick summary of results
