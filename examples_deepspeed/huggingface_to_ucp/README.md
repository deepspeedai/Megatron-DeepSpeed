# Converting Hugging Face Checkpoints to Universal Checkpointing Format

This folder demonstrates the correctness of the `hf_to_universal.py` script provided in the DeepSpeed repository. Below is a four-step process to show that `hf_to_universal.py` script correctly converts Hugging Face checkpoints to Universal Checkpointing format and that resuming training with the (converted) universal checkpoint results in the same loss curve as the original ZeRO checkpoint.

1. ZeRO-based training run, optionally combining TP and PP or SP, that creates normal ZeRO checkpoints.  
2. Converting ZeRO checkpoint into Huggingface (pytorch_model.bin or safetensors) format using `zero_to_fp32.py` utility of DeepSpeed.
3. **Converting Huggingface checkpoint into the universal format using `hf_to_universal.py` utility of DeepSpeed.**
4. Resuming training with the universal checkpoint, on a different parallelism topologies. 

### Download and Pre-process Training Dataset
Before executing the steps below, you can download and pre-process the training set using the following commands (see [here](https://github.com/bigscience-workshop/Megatron-DeepSpeed?tab=readme-ov-file#quick-pre-processing-to-start-training-with) for more details):
```bash
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
xz -d oscar-1GB.jsonl.xz
python tools/preprocess_data.py \
    --input oscar-1GB.jsonl \
    --output-prefix my-gpt2 \
    --vocab-file gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8
```

NOTE: Make sure to update your `BASE_DATA_PATH` path in the `run_[bf16/fp16].sh` and `run_universal_[bf16/fp16].sh` scripts to point to the pre-processed data.

## Step 1: ZeRO stage 3 training
Follow [Step 1 in Universal Checkpointing guide](https://github.com/deepspeedai/Megatron-DeepSpeed/tree/main/examples_deepspeed/universal_checkpointing#step-1-create-zero-checkpoint) 

For example, run the following command:
```bash
ZERO_STAGE=3 DP=4 bash examples_deepspeed/universal_checkpointing/megatron_gpt/run_bf16.sh
```

After training for 100 iterations, you should have a ZeRO checkpoint in your checkpoint directory, for example `z3_uni_ckpt/checkpoints/gpt2/z1/bf16/tp2_pp2_dp2_sp1_toy/global_step100`.

## Step 2: Convert ZeRO checkpoint to Hugging Face format
In your checkpoint directory, for example `z3_uni_ckpt/checkpoints/gpt2/z1/bf16/tp2_pp2_dp2_sp1_toy/global_step100`, run the following command:

```bash
python zero_to_fp32.py \
    --checkpoint_dir . \
    --output_dir global_step100_hf \
```

## Step 3: Convert Hugging Face checkpoint to Universal format
In your checkpoint directory, for example `z3_uni_ckpt/checkpoints/gpt2/z1/bf16/tp2_pp2_dp2_sp1_toy/global_step100_hf`, run the following command:

```bash
python ${HOME}/DeepSpeed/deepspeed/checkpoint/hf_to_universal.py \
    --hf_checkpoint_dir z3_uni_ckpt/checkpoints/gpt2/z1/bf16/tp2_pp2_dp2_sp1_toy/global_step100_hf \
    --save_dir z3_uni_ckpt/checkpoints/gpt2/z1/bf16/tp2_pp2_dp2_sp1_toy/global_step100_hf_universal
```

## Step 4: Resume training with Universal checkpoint
Follow [Step 3 in Universal Checkpointing guide](https://github.com/deepspeedai/Megatron-DeepSpeed/tree/main/examples_deepspeed/universal_checkpointing#step-3-resume-training-with-universal-checkpoint-of-iteration-100)

Note that the flag `IGNORE_MISSING_OPTIM_LOWER` is set to `true` to ignore the missing optimizer state in the Universal Checkpoint.

For example, run the following command:
```bash
ZERO_STAGE=3 DP=4 LOAD_DP=4 IGNORE_MISSING_OPTIM_LOWER=true bash examples_deepspeed/universal_checkpointing/megatron_gpt/run_universal_bf16.sh
```

### TensorBoard Log Analysis

The Universal Checkpointing example in the `universal_checkpointing` folder includes a TensorBoard analysis script that will generate `csv` files and `png` plots across the unviersal checkpointing training steps for comparison of training and validation loss curves.

After Step 4 is completed, the script may be executed as follows:
```bash
bash examples_deepspeed/universal_checkpointing/megatron_gpt/run_tb_analysis_gpt.sh z3_uni_ckpt
```
Below is the visualization of the `png` files generated from this example.

<div align="center">
  <img src="assets/image/uc_char_training_loss_z3_tp1_pp1_dp4_sp1_mbsz4" alt="" width="600"/>
  *Figure 1: Training LM loss curve for first 200 training steps of Step 1 (DP=4) and training steps 101 to 200 of Step 4 (DP=4), which was loaded using the Universal Checkpoint.*
</div>

<div align="center">
  <img src="assets/image/uc_char_validation_loss_z3_tp1_pp1_dp4_sp1_mbsz4" alt="" width="600"/>
  *Figure 2: Validation LM loss curve for first 200 training steps of Step 1 (DP=4) and training steps 101 to 200 of Step 4 (DP=4), which was loaded using the Universal Checkpoint.*
</div>




