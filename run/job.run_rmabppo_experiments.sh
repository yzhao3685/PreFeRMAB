#!/bin/bash

# make dir to store experiment outputs
mkdir -p expts
counter=0
while [[ -e expts/output_${counter}.txt ]]; do
    ((counter++))
done
echo -e "\e[1;92m==== Starting Experiments, Logging @ expts/output_${counter}.txt ====\e[0m"
exec &> expts/output_${counter}.txt

# run the entire script 10 times
for main_run in {1..1}; do

  echo -e "\e[1;92m==== Starting Main Run $main_run ====\e[0m"
  rm -rf ./data/*

  # Experiment parameters to search
  N_values=(21) #(21 48 96)
  B_values=(7.0) #(7.0 16.0 32.0)
  opt_in_rates=(0.8 0.85 0.9 0.95 1.0)

  # Fixed parameters
  data="counterexample"
  save_string="ce_rmabppo_test"
  robust_keyword="sample_random"
  n_train_epochs=100
  seed=0
  cdir="."
  no_hawkins=1
  tp_transform="linear"
  data_type="discrete" #only for approximation tests, can ignore
  agent_tp_transform_dims=4
  training_opt_in_rate=0.8

  for i in "${!N_values[@]}"; do
    N=${N_values[$i]}
    B=${B_values[$i]}
    echo -e "\e[1;31m==== STARTING EXPERIMENT: N=$N, B=$B ====\e[0m"
    bash run/run_rmabppo_experiment_train.sh $cdir $seed 0 $data $save_string $N $B \
                                            $robust_keyword $n_train_epochs $no_hawkins $tp_transform $training_opt_in_rate $data_type $agent_tp_transform_dims
    for opt_in_rate in "${opt_in_rates[@]}"; do
      for run in {1..1}; do
        echo -e "\e[1;34m==== RUN $run FOR opt_in_rate=$opt_in_rate ====\e[0m"
        bash run/run_rmabppo_experiment_test.sh $cdir $seed 0 $data $save_string $N $B \
                                                $robust_keyword $n_train_epochs $no_hawkins $tp_transform $opt_in_rate $data_type
      done
    done
  done
done
