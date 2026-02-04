#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

method="TAPPA" #KV cache method: FullKV, Cake, TAPPA, PyramidKV, SnapKV, H2O, StreamingLLM
model_name="llama3.1-8b-128k"
max_capacity_prompts=1024 # KV cache compression budget
window_size=32 # for computing compress metrics
alpha="inf" # alpha in TAPPA paper (controls q similarity ratio, 'inf' means only use q-similarity)
pred_name="tappa_alpha_inf" # experiment name for output file

tasks=("narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique" "qmsum" "triviaqa" "passage_retrieval_en" "lcc" "repobench_p" "gov_report" "multi_news" "trec" "samsum" "passage_count")

# tasks=("multifieldqa_en") # you can also run a single task for debugging purposes

python KVCache/pred_kvcache.py \
    --method "${method}" \
    --model "${model_name}" \
    --cascading \
    --pred_name "${pred_name}" \
    --device 0 \
    --max_capacity_prompts "${max_capacity_prompts}" \
    --window_size "${window_size}" \
    --task "${tasks[*]}" \
    --alpha "${alpha}" 

