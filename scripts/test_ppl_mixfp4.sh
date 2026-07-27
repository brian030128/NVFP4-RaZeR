#!/bin/bash

########## Modify the path according to your HOME directory ##########
HOME_DIR="/home/yc2367/llm/NVFP4-RaZeR"
######################################################################

# MixFP4 type-block sweep. The scale block is always the 16-element NVFP4 block; only the
# type block (which selects E2M1 vs E0M3) changes. The K dimension of a type block must be a
# multiple of 16. Results are written to one file per type-block shape.

seq_len=2048
OUTPUT_DIR=${HOME_DIR}/results/ppl_${seq_len}
dataset_list="wikitext,c4"

model_list=(
    "llama-2-7b" "llama-3.1-8b" "qwen3-8b"
)

w_bits=4
w_groupsize=16
a_bits=4
a_groupsize=16

type_block_list=("1x16" "16x16" "256x16" "32x64" "32x128")

for model_name in "${model_list[@]}"
do
    # NVFP4 reference point
    python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
        --datasets ${dataset_list} --seq_len ${seq_len} \
        --output_dir ${OUTPUT_DIR} \
        --w_bits ${w_bits} --w_groupsize ${w_groupsize} --w_dtype nvfp4 \
        --a_bits ${a_bits} --a_groupsize ${a_groupsize} --a_dtype nvfp4

    for type_block in "${type_block_list[@]}"
    do
        # weight-only MixFP4
        python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
            --datasets ${dataset_list} --seq_len ${seq_len} \
            --output_dir ${OUTPUT_DIR} \
            --w_bits ${w_bits} --w_groupsize ${w_groupsize} --w_dtype mixfp4 --w_type_block ${type_block} \
            --a_bits 16 --a_dtype fp16

        # W4A4 MixFP4
        python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
            --datasets ${dataset_list} --seq_len ${seq_len} \
            --output_dir ${OUTPUT_DIR} \
            --w_bits ${w_bits} --w_groupsize ${w_groupsize} --w_dtype mixfp4 --w_type_block ${type_block} \
            --a_bits ${a_bits} --a_groupsize ${a_groupsize} --a_dtype mixfp4 --a_type_block ${type_block}
    done
done
