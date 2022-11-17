#!/bin/bash
dir=`pwd`
###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
seq_len=2048

### The "GPT-3 XXX" below are configs from GPT-3 paper
### https://arxiv.org/abs/2005.14165, choose based on
### your desired model size or build your own configs

## init_std is standard deviation for weight initialization. Usually larger
## model needs lower std. We used a heuristic equation of sqrt(1/3/hidden_size)
## from the MT-NLG 530B work (https://arxiv.org/pdf/2201.11990.pdf)

## GPT-3 Small 125M
model_size=0.125
num_layers=12
hidden_size=768
num_attn_heads=12
global_batch_size=256
lr=6.0e-4
min_lr=6.0e-5
init_std=0.02

## GPT-3 Medium 350M
# model_size=0.35
# num_layers=24
# hidden_size=1024
# num_attn_heads=16
# global_batch_size=256
# lr=3.0e-4
# min_lr=3.0e-5
# init_std=0.018

## GPT-3 Large 760M
# model_size=0.76
# num_layers=24
# hidden_size=1536
# num_attn_heads=16
# global_batch_size=256
# lr=2.5e-4
# min_lr=2.5e-5
# init_std=0.015

## GPT-3 XL 1.3B
# model_size=1.3
# num_layers=24
# hidden_size=2048
# num_attn_heads=16
# global_batch_size=512
# lr=2.0e-4
# min_lr=2.0e-5
# init_std=0.013

## GPT-3 2.7B
# model_size=2.7
# num_layers=32
# hidden_size=2560
# num_attn_heads=32
# global_batch_size=512
# lr=1.6e-4
# min_lr=1.6e-5
# init_std=0.011

## GPT-3 6.7B
# model_size=6.7
# num_layers=32
# hidden_size=4096
# num_attn_heads=32
# global_batch_size=1024
# lr=1.2e-4
# min_lr=1.2e-5
# init_std=0.009

## GPT-3 13B
# model_size=13
# num_layers=40
# hidden_size=5120
# num_attn_heads=40
# global_batch_size=1024
# lr=1.0e-4
# min_lr=1.0e-5
# init_std=0.008

## GPT-3 175B
# model_size=175
# num_layers=96
# hidden_size=12288
# num_attn_heads=96
# global_batch_size=1536
# lr=0.6e-4
# min_lr=0.6e-5
# init_std=0.005
###############################################################################
### Training duration configs
## To strictly follow original GPT-3 paper, this script uses the batch size
## warmup technique which "gradually increase the batch size linearly from a
## small value (32k tokens) to the full value over the first 4-12 billion
## tokens of training, depending on the model size". In order to support the
## small batch sizes, the training need to be done in two stages.
bsz_warmup_tokens_in_billion=4
bsz_warmup_samples=$((${bsz_warmup_tokens_in_billion} * 1000000000 / ${seq_len}))
bsz_warmup_schedule="16 16 ${bsz_warmup_samples}"

## Set this to true if you resume from an unfinished training but the first
## stage already finished
resume_from_second_stage="false"
# resume_from_second_stage="true"

## The main termination condition, original GPT-3 paper trains for 300B tokens.
train_tokens_in_billion=300
train_tokens=$((${train_tokens_in_billion} * 1000000000))
## Below is num token for first batch size warmup stage only. This number is
## 5M larger than the actual num warmup token to make sure we fully finish
## the warmup in first stage.
train_tokens_1st_stage=$((${bsz_warmup_tokens_in_billion} * 1000000000 + 5000000))

## train_samples is another termination condition and also affect the number of 
## data samples to be indexed. Since we want to reach the train_tokens
## above, and data efficiency techniques may change num tokens in some samples,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by train_samples.
train_samples=$(( 300 * 1000000000 * 2 / ${seq_len} ))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
exit_duration=30000000
###############################################################################
### lr configs
## lr warmup and decay duration.
## Original GPT-3 paper uses 375M warmup tokens and 260B cosine decay tokens.
lr_warmup_tokens_in_million=375
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))
lr_decay_tokens_in_billion=260
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000000000))
lr_decay_style="cosine"
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
batch_size=4
## This is the micro batch size for first batch size warmup stage only. For
## first stage the total number of gpus has to be no more than 16, and the
## micro batch size has to be 16/numgpu since the batch starts with 16 and
## increment by multiple of 16.
batch_size_1st_stage=1

## Model parallelism, 1 is no MP
mp_size=1

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
pp_size=1
no_pp="true"

## ZeRO stage
zero_stage=0

## Total number of GPUs. ds_ssh is from DeepSpeed library.
num_gpus=$(($(ds_ssh nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)-2))
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node=$(( ${num_gpus} / ${num_gpus_pernode} ))
## Because 1st stage starts warmup batch size from 16 with a step of 16, we can
## only use up to 16 gpus.
num_node_1st_stage=$(( 16 / ${num_gpus_pernode} ))
num_node_1st_stage=$(( ${num_node_1st_stage} > 1 ? ${num_node_1st_stage} : 1 ))
num_gpus_pernode_1st_stage=$(( ${num_gpus_pernode} < 16 ? ${num_gpus_pernode} : 16 ))
## Data parallel size. Currently not used as any config, just for record.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))
###############################################################################
### Misc configs
log_interval=100
eval_iters=10
eval_interval=100
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=20
estimated_train_iter=$((${train_tokens} / ${seq_len} / ${global_batch_size}))
save_interval=$((${estimated_train_iter} / ${num_save}))

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
seed=1234

use_internal_data="false"
if [ "${use_internal_data}" = "true" ]; then
    jobname="gpt-internal"
    ## The internal data is only accessible within Microsoft
    data_home="/vc_data_blob/users/conglli/pile-cc1-cc2-shuf"
    if [[ "$host" == *"webxt"* ]]; then
        data_home="/blob/data/pile-cc1-cc2-shuf"
    fi
    arx="${data_home}/ArXiv_ftfy_cleaned_id_shuf_text_document"
    bc2="${data_home}/BookCorpus2_ftfy_cleaned_id_shuf_text_document"
    b3="${data_home}/Books3_ftfy_cleaned_id_shuf_text_document"
    cc2020="${data_home}/CC-2020-50_id_cleaned_shuf_text_document"
    cc2021="${data_home}/CC-2021-04_id_cleaned_shuf_text_document"
    git="${data_home}/Github_ftfy_id_shuf_text_document"
    gut="${data_home}/Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document"
    nih="${data_home}/NIH_ExPorter_ftfy_id_shuf_text_document"
    owt2="${data_home}/OpenWebText2_ftfy_cleaned_id_shuf_text_document"
    pcc="${data_home}/Pile-CC_id_cleaned_shuf_text_document"
    pm="${data_home}/PubMed_Abstracts_ftfy_id_shuf_text_document"
    rn="${data_home}/rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document"
    se="${data_home}/StackExchange_ftfy_id_shuf_text_document"
    st="${data_home}/stories_dedup0.7_shuf_cleaned_shuf_text_document"
    wik="${data_home}/Wikipedia_en_ftfy_id_shuf_text_document"
    data_path="0.14336 ${b3} 0.08962 ${rn} 0.19336 ${owt2} 0.05689 ${se} \
    0.00859 ${st} 0.02897 ${pm} 0.04771 ${wik} 0.00873 ${gut} 0.01007 ${bc2} \
    0.00208 ${nih} 0.13017 ${cc2020} 0.09446 ${pcc} 0.15652 ${cc2021} \
    0.01359 ${arx} 0.01588 ${git}"
    override_data_index="false"
else
    jobname="gpt-pile"
    ## Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
    data_home="/vc_data_blob/users/conglli/the_pile_public_merged_nopreprocessing"
    if [[ "$host" == *"webxt"* ]]; then
        data_home="/blob/data/the_pile_public_merged_nopreprocessing"
    fi
    data_path="${data_home}/pile_text_document"
fi

vocab_path="gpt2-vocab.json"
if [ ! -f "$vocab_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi
merge_path="gpt2-merges.txt"
if [ ! -f "$merge_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
fi

jobname="${jobname}-${model_size}B-tok${train_tokens_in_billion}B"
jobname="${jobname}-lr${lr}-min${min_lr}-wup${lr_warmup_tokens_in_million}M-dcy${lr_decay_tokens_in_billion}B-sty-${lr_decay_style}"
jobname="${jobname}-gbs${global_batch_size}-mbs${batch_size}-gpu${num_gpus}-zero${zero_stage}-mp${mp_size}-pp${pp_size}"
if [ "${no_pp}" = "true" ]; then
    jobname="${jobname}-nopp"
fi
jobname="${jobname}-seed${seed}"
jobname="${jobname}-bwup${bsz_warmup_tokens_in_billion}B"

username=$(whoami)
output_home="/blob/users/${username}/project/data_efficient_gpt"
log_path="${output_home}/log/"
checkpoint_path="${output_home}/checkpoint/${jobname}"
## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
tensorboard_dir="/data/users/${username}/project/data_efficient_gpt/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
###############################################################################
data_options=" \
    --vocab-file ${vocab_path} \
    --merge-file ${merge_path} \
    --data-path ${data_path} \
    --data-impl mmap"
        
megatron_options=" \
    --override-lr-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${mp_size} \
    --init-method-std ${init_std} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-tokens ${lr_warmup_tokens} \
    --exit-duration-in-mins ${exit_duration} \
    --rampup-batch-size ${bsz_warmup_schedule} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-samples ${train_samples} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers 0 \
    --fp16 \
    --seed ${seed} \
    --load ${checkpoint_path} \
    --save ${checkpoint_path} \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard"

megatron_options_bszwarmup=" \
    --micro-batch-size ${batch_size_1st_stage} \
    --train-tokens ${train_tokens_1st_stage} \
    --tensorboard-dir ${tensorboard_path}"

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

template_json="ds_config_gpt_TEMPLATE.json"
config_json="ds_config_gbs${global_batch_size}_mbs${batch_size_1st_stage}_log${log_interval}_zero${zero_stage}.json"
if [[ $zero_stage -gt 0 ]]; then
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size_1st_stage}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/false/" \
      > ${config_json}
else
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size_1st_stage}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/true/" \
      > ${config_json}
fi

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}"

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi

if [ "${resume_from_second_stage}" = "false" ]; then
    ## When saving checkpoint to a storage with cache, their could be consistency
    ## issue of the pointer to latest checkpoint. Here we find the correct pointer
    ## and broadcast it to all nodes.
    iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
    iteration_file_2="$checkpoint_path/latest"
    iteration=0
    for (( node = 0; node <= num_node-1; node++ ))
    do
        if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
            local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
            iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} :  ${iteration} ))
        fi
    done
    if [[ $iteration -gt 0 ]]; then
        iteration_2="global_step${iteration}"
        ds_ssh "echo $iteration > $iteration_file"
        ds_ssh "echo $iteration_2 > $iteration_file_2"
    fi

    deepspeed --num_nodes $num_node_1st_stage --num_gpus $num_gpus_pernode_1st_stage ${dir}/../../../../pretrain_gpt.py ${megatron_options} ${megatron_options_bszwarmup} ${data_options} ${deepspeed_options} &> ${log_path}/${jobname}_${host}_${current_time}.log
    sleep 120s
fi

## Full batch size second stage
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"

megatron_options_bszwarmup=" \
        --micro-batch-size ${batch_size} \
        --train-tokens ${train_tokens} \
        --tensorboard-dir ${tensorboard_path}"

template_json="ds_config_gpt_TEMPLATE.json"
config_json="ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}.json"
if [[ $zero_stage -gt 0 ]]; then
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/false/" \
      > ${config_json}
else
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/true/" \
      > ${config_json}
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0
for (( node = 0; node <= num_node-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
        local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
        iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} :  ${iteration} ))
    fi
done
if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    ds_ssh "echo $iteration > $iteration_file"
    ds_ssh "echo $iteration_2 > $iteration_file_2"
fi

deepspeed ${dir}/../../../../pretrain_gpt.py ${megatron_options} ${megatron_options_bszwarmup} ${data_options} ${deepspeed_options} &>> ${log_path}/${jobname}_${host}_${current_time}.log