## CAUTION: first read Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md
## and follow the steps of installation/data downloading.
checkpoint_paths=(
    /vc_data_blob/users/conglli/project/data_efficient_gpt/checkpoint/gpt-pile-0.125B-tok300B-lr6.0e-4-min6.0e-5-wup375M-dcy260B-sty-cosine-gbs256-mbs4-gpu64-zero0-mp1-pp1-nopp-seed1234-bwup4B/global_step591581/
    /vc_data_blob/users/conglli/project/data_efficient_gpt/checkpoint/gpt-pile-0.125B-tok300B-lr6.0e-4-min6.0e-5-wup3000M-dcy260B-sty-cosine-gbs256-mbs4-gpu64-zero0-mp1-pp1-nopp-seed1234/global_step572205/
)

## No need to use the exact training config json, just use this dummy is fine
config_path=ds_config_eval_dummy.json
username=$(whoami)
result_path="/blob/users/${username}/project/data_efficient_gpt/eval_results_fewshot"

## Task(s) on the same row will be performed together in the same process.
tasks=(
    record # num_fewshot=32
    triviaqa # num_fewshot=64
    hellaswag # num_fewshot=20
    arc_challenge # num_fewshot=50
    arc_easy # num_fewshot=50
    race # num_fewshot=10
    multirc # num_fewshot=32
    openbookqa # num_fewshot=100
    lambada # num_fewshot=15
    webqs # num_fewshot=64
    winogrande # num_fewshot=50
    piqa # num_fewshot=50
    anli_r1,anli_r2 # num_fewshot=50
    anli_r3 # num_fewshot=50
    boolq,copa # num_fewshot=32
    rte,wsc # num_fewshot=32
)

num_fewshots=(
    32
    64
    20
    50
    50
    10
    32
    100
    15
    64
    50
    50
    50
    50
    32
    32
)

## Use localhost if you didn't setup hostfile as described in
## https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node.
## If hostfile exist, use hostname (e.g., worker-0) in hostfile.
# hostname="localhost"
hostname="worker-0"

batch_size=16

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
cuda_id=-1
total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv -i 0 | grep -Eo [0-9]+)

## Code below only works when you run each evalharness task on a single GPU.
## For multi-GPU evalharness, check Megatron-DeepSpeed/blob/main/examples/MoE/ds_evalharness.sh
for l in "${!checkpoint_paths[@]}"; do 
    checkpoint_path=${checkpoint_paths[l]}
    for ((i=0;i<${#tasks[@]};++i)); do
        task=${tasks[i]}
        num_fewshot=${num_fewshots[i]}
        free_mem=0
        while [ $free_mem -lt $total_mem ]; do
            cuda_id=$(((cuda_id+1)%num_gpus))
            free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $cuda_id | grep -Eo [0-9]+)
            sleep 60s
        done
        bash ds_evalharness_1gpu.sh $checkpoint_path $config_path $result_path $cuda_id $task $hostname $batch_size $num_fewshot &
    done
done
