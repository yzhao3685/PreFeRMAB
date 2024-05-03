python3 ${1}/agent_oracle.py --hid 32 -l 2 --gamma 0.9 --cpu 1 \
    --exp_name ${5} \
    --home_dir ${1} \
    -s ${2} \
    --cannon ${3} \
    --data ${4} \
    --save_string ${5} \
    -N ${6} -B ${7} \
    --opt_in_rate ${12} \
    --data_type ${13} \
    --agent_steps 100 \
    --agent_epochs ${9} \
    --agent_init_lambda_trains 0 \
    --agent_clip_ratio 2 \
    --agent_final_train_lambdas 20 \
    --agent_start_entropy_coeff 0.5 \
    --agent_end_entropy_coeff 0 \
    --agent_pi_lr 2e-3 \
    --agent_vf_lr 2e-3 \
    --agent_lm_lr 2e-3 \
    --agent_train_pi_iters 20 \
    --agent_train_vf_iters 20 \
    --agent_lamb_update_freq 4 \
    --robust_keyword ${8} \
    --agent_tp_transform ${11} \
    --agent_tp_transform_dims ${14} \
    --scheduler_discount 0.95 \
