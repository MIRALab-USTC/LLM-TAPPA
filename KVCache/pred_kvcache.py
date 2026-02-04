import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
from datetime import datetime, timedelta
import time
from pathlib import Path

from cake.cake_cache import CakeprefillKVCache
from cake.utils import CompressConfig


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama3.1-8b-128k",
                        choices=["llama3.1-8b-128k", "qwen2.5-7b-instruct"])
    parser.add_argument('--method', type=str, default="fullkv",
                        help="KV cache method: FullKV, Cake, TAPPA, PyramidKV, SnapKV, H2O, StreamingLLM")
    
    parser.add_argument('--pred_name', type=str, default="pred", help="Name for output file")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", choices=["flash_attention_2"])
    parser.add_argument("--task", type=str, default=None, help="Datasets to evaluate on, separated by space.")

    # KV Cache compression
    parser.add_argument('--max_capacity_prompts', type=int, default=1024)
    parser.add_argument('--window_size', type=int, default=32)

    # Cake (default in Cake paper)
    parser.add_argument('--cascading', action='store_true', help="Default True in Cake. Using cascading cache management")
    parser.add_argument('--tau1', type=float, default=1.0)
    parser.add_argument('--tau2', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=200.0)

    # TAPPA
    parser.add_argument('--alpha', type=float, default=float("inf"),
                        help="Alpha in TAPPA paper (controls q similarity ratio, can be inf)")

    # Methods in KV-Factory (PyramidKV, SnapKV, H2O, StreamingLLM...)
    parser.add_argument('--merge', type=str, default=None, help='KV merge method for PyramidKV')
    parser.add_argument('--floor', type=float, default=0.2, help='AdaKV floor')
    parser.add_argument('--pruning_ratio', type=float, default=0.4, help='pruning ratio used in some methods')
    parser.add_argument('--recent_size', type=int, default=32)
    
    # seed
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--do_sample', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--temperature', type=float, default=0.1)

    args = parser.parse_args(args)
    return args

def build_chat(tokenizer, prompt, model_name):
    if "llama3" in model_name:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name:
        prompt = f'<s>[INST] {prompt} [/INST]'
    elif "qwen" in model_name:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt



@torch.inference_mode()
def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, device, args):
    fout = open(out_path, "w")
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench_p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=args.do_sample,
                temperature=args.temperature,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=args.do_sample,
                temperature=args.temperature,
            )[0]


        method = args.method.lower() if args.method is not None else "fullkv"
        if method in ["cake", "tappa"]:
            layers = len(model.model.layers)
            for i in range(layers):
                model.model.layers[i].self_attn.config.prefill = [True]*layers
                model.model.layers[i].self_attn.config.decoding_evict = [None]*layers

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        fout.write(json.dumps({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, ensure_ascii=False) + "\n")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, args):
    # monkeypatch for Cake, PyramidKV, etc.
    method = args.method.lower() if args.method is not None else "fullkv"
    if method in ["pyramidkv", "snapkv", "h2o", "streamingllm"]:
        from pyramidkv.monkeypatch import replace_llama, replace_mistral, replace_qwen2
        replace_llama(method)
        replace_mistral(method)
        replace_qwen2(method)
    elif method == "cake":
        if "llama" in model_name:
            from cake.monkeypatch import replace_flashllama_attn_with_cakeattn
            replace_flashllama_attn_with_cakeattn()
        elif "mistral" in model_name:
            from cake.monkeypatch import replace_flashmistral_attn_with_cakeattn
            replace_flashmistral_attn_with_cakeattn()
        elif "qwen2" in model_name:
            from cake.monkeypatch import replace_flashqwen2_attn_with_cakeattn
            replace_flashqwen2_attn_with_cakeattn()
        else:
            raise ValueError(f"CAKE does not support model: {model_name}")
    elif method == "tappa":
        from tappa.patch import apply_tappa_patch
        if "llama" in model_name:
            apply_tappa_patch("llama")
        elif "qwen2" in model_name:
            apply_tappa_patch("qwen2")
        else:
            raise ValueError(f"TAPPA does not support model: {model_name}")
    elif method == "fullkv":
        pass
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    if "qwen2" in model_name:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype,
        attn_implementation=args.attn_implementation
    ).to(device)
    if 'llama-3' in path.lower():
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.top_k = None
    model.generation_config.top_p = None
    config = AutoConfig.from_pretrained(path)
    if hasattr(config, 'num_hidden_layers'):
        layers = config.num_hidden_layers

    # CAKE / TAPPA config
    if method in ["cake", "tappa"]:
        compress_config = CompressConfig(compress=True, cascading=args.cascading)
        compress_config.max_capacity_prompts = args.max_capacity_prompts
        compress_config.window_size = args.window_size
        model2tau = json.load(open("config/model2tau_cake.json", "r"))
        try:
            tau1 = model2tau[model_name][f"{args.max_capacity_prompts}"]["tau1"]
            tau2 = model2tau[model_name][f"{args.max_capacity_prompts}"]["tau2"]
            print(tau1, tau2)
        except Exception as e:
            print(f"Error loading tau values: {e}")
            tau1, tau2 = 1.0, 1.0
        gamma = args.gamma
        compress_config.hyper = [tau1, tau2, gamma]

        for i in range(layers):
            attn_cfg = model.model.layers[i].self_attn.config
            attn_cfg.key_size = [compress_config.max_capacity_prompts - compress_config.window_size]*layers
            attn_cfg.window_size = [compress_config.window_size]*layers
            attn_cfg.prefill = [True]*layers
            attn_cfg.decoding_evict = [None]*layers
            attn_cfg.tau1 = compress_config.hyper[0]
            attn_cfg.tau2 = compress_config.hyper[1]
            attn_cfg.gamma = compress_config.hyper[2]
            attn_cfg.prefill_cake_evict = [CakeprefillKVCache(
                cache_size=compress_config.max_capacity_prompts,
                window_size=compress_config.window_size,
                k_seq_dim=2, v_seq_dim=2,
                num_heads=model.model.layers[i].self_attn.num_heads,
                num_layers=layers,
                use_cascading=compress_config.cascading
            )]*layers

            if method == "tappa":
                attn_cfg.alpha = args.alpha

    # PyramidKV/SnapKV/StreamingLLM methods
    if method in ["pyramidkv", "snapkv", "h2o", "streamingllm"]:
        window_size = args.window_size
        max_capacity_prompts = args.max_capacity_prompts
        kernel_size = 7  # default
        pooling = "maxpool"
        ratio = args.pruning_ratio
        recent_size = args.recent_size
        merge = args.merge
        floor = args.floor
        for i in range(layers):
            attn_cfg = model.model.layers[i].self_attn.config
            attn_cfg.window_size = window_size
            attn_cfg.max_capacity_prompt = max_capacity_prompts
            attn_cfg.kernel_size = kernel_size
            attn_cfg.pooling = pooling
            attn_cfg.merge = merge
            attn_cfg.floor = floor
            attn_cfg.ratio = ratio
            attn_cfg.recent_size = recent_size
    model = model.eval()



if __name__ == '__main__':
    args = parse_args()
    print(args)
    seed_everything(args.seed)
    pred_name = args.pred_name
    model_name = args.model
    method = args.method.lower() if args.method is not None else "fullkv"
    args.method = method
    
    # Load configurations
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    max_length = model2maxlen[model_name]
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model_path = model2path[model_name]

    # Use timestamp + PID so repeated runs don't overwrite previous results
    suffix = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if method in ["cake", "tappa", "pyramidkv", "snapkv", "h2o", "streamingllm"]:
        cache_name = f"cache{args.max_capacity_prompts}"
    else:
        cache_name = "cachefull"
    save_dir = f"./pred_result/{cache_name}/{pred_name}/seed{args.seed}/{suffix}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_path = os.path.join(save_dir, model_name)
    model, tokenizer = load_model_and_tokenizer(model_path, model_name, device, args)
    # Support multiple tasks (space separated)
    datasets = args.task.split()
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    start_time = datetime.now()
    for idx, dataset in enumerate(datasets):
        print(f"Working on dataset {dataset} - {idx}/{len(datasets)}")
        data_files = {"test": f"{dataset}.jsonl"}
        data = load_dataset("json", data_dir='data/LongBench', split='test', data_files=data_files)
        out_path = os.path.join(output_path, f"{dataset}.jsonl")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        get_pred(model, tokenizer, data_all, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, device, args)
        duration = datetime.now() - start_time
        print(f"time after start is {str(timedelta(seconds=duration.total_seconds()))}.")
    # evaluation
    cmd = f"python eval.py --model {args.model} --cache_size {args.max_capacity_prompts} --eval_avg --dir_path {output_path}"
    print(cmd)
    os.system(cmd)
