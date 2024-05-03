data="continuous_state"
save_string="ce_rmabppo_test"
N=21
B=7.0
robust_keyword="sample_random" # other option is "mid"
n_train_epochs=30 # 10 epoch is ok 
seed=0
cdir="."
no_hawkins=1
tp_transform=None
opt_in_rate=0.9
data_type="discrete"

bash run/synthetic_continuous/run_rmabppo_continuous_state.sh ${cdir} ${seed} 0 ${data} ${save_string} ${N} ${B}  \
    ${robust_keyword} ${n_train_epochs} ${no_hawkins} ${tp_transform} ${opt_in_rate} ${data_type}



