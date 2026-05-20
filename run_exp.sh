#!/bin/bash
mkdir -p logs
source ~/quantum-sat-env/bin/activate
cd /mnt/d/Quantum_kernel/quantum-sat-classification
PYTHONPATH=. nohup python experiments/exp1_physics.py > logs/exp1_physics_bw050.log 2>&1 &
PID=$!
echo $PID > run_pid.txt
echo "PID: $PID"
