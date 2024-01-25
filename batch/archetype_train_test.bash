#!/bin/bash

# Save a backup of the original stdout and stderr file descriptors
exec 3>&1 4>&2

# Redirect stdout and stderr to a file
export LOG_PATH="/scratch/bf996/notebooks/archetype/logs"
cur_time=$(date +%s)

log_file="${LOG_PATH}/archetype_${cur_time}.log"
exec &> "$log_file"

main_filename=archetypellama-cont+resam-full+oc-15sample.json

export JSON_OUT_PATH="/scratch/bf996/llm_er_std/proj/CTA_CPA_Benchmarks/wotab/${main_filename}"

#filename=$(basename "$JSON_OUT_PATH")
filename=$(echo $JSON_OUT_PATH | sed 's/.....$//')

suffix="-dataset.json"
new_filename="${filename}${suffix}"

# Replace the original filename with the new filename in the path
export FT_DATA_PATH="$new_filename"

suffix2="-ft-results.json"
ft_filename="${filename}${suffix}"

export FT_RESULTS_PATH="$ft_filename"

export GEN_MODEL_PATH="/scratch/bf996/text-generation-webui/models/llama-7b-cta-oc"
export BASE_WEIGHTS="/scratch/bf996/text-generation-webui/models/llama-7b-alpaca"
export OUTPUT_WEIGHTS="/scratch/bf996/text-generation-webui/models/llama-7b-cta-oc-new"
export ADDL_FLAGS=" --other_col --table_src --summ_stats --sample_size=15 "
export FT_ADDL_FLAGS="--num_train_epochs 3 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1"


cd /scratch/bf996/notebooks

sed -e "s#%%GEN_MODEL_PATH%%#${GEN_MODEL_PATH}${ADDL_FLAGS}#g" \
    -e "s#%%JSON_OUT_PATH%%#${JSON_OUT_PATH}#g" \
    archetype/batch/make_archetype_dataset.batch > archetype/batch/make_archetype_dataset_mod.batch

job1=$(sbatch archetype/batch/make_archetype_dataset_mod.batch | awk '{print $4}')

echo "Executing job 1"

echo $job1

cd /scratch/bf996/stanford_alpaca

sed -e "s#%%BASE_WEIGHTS%%#${BASE_WEIGHTS}#g" \
    -e "s#%%FT_DATA_PATH%%#${FT_DATA_PATH}#g" \
    -e "s#%%OUTPUT_WEIGHTS%%#${OUTPUT_WEIGHTS}#g" \
    -e "s#%%ADDL_FLAGS%%#${FT_ADDL_FLAGS}#g" \
    batch/finetune_llama_ctabase.sbatch > batch/finetune_llama_ctabase_mod.sbatch

job2=$(sbatch --dependency=afterok:$job1 batch/finetune_llama_ctabase_mod.sbatch | awk '{print $4}')

echo "Executing job 2"

echo $job2

cd /scratch/bf996/notebooks

sed -e "s#%%FT_RESULTS_PATH%%#${FT_RESULTS_PATH}${ADDL_FLAGS}#g" \
    -e "s#%%TEST_MODEL%%#${OUTPUT_WEIGHTS}#g" \
    archetype/batch/test_archetype_sotab_ft.batch > archetype/batch/test_archetype_sotab_ft_mod.batch

job3=$(sbatch --dependency=afterok:$job2 archetype/batch/test_archetype_sotab_ft_mod.batch)

echo "Executing job 3"

echo $job3

echo "Done"

exec 1>&3 2>&4