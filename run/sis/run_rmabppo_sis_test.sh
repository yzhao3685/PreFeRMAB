exp_name=${5}_n${6}b${7}d${4}r${8}p${11}
python3 ${1}/robust_rmab/simulator.py --discount 0.9 \
--budget ${7} \
 --data ${4} \
-N ${6} \
-s ${2} -ws ${2} \
-rlmfr ${1}/data/${exp_name}/${exp_name}_s${2}/ \
-L 10 \
 -n 50 \
 --robust_keyword ${8} \
 --file_root ${1} \
 --save_string ${5} \
 --no_hawkins ${10} \
 --opt_in_rate ${12} \
 --pop_size ${11} 