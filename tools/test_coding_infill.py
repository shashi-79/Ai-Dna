import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import eval_code_submission, generate_coding_subset

def main():
    templates = generate_coding_subset(10)
    tok = AutoTokenizer.from_pretrained('modal/text_models/qwen2.5-0.5b')
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained('my_llm_folder', torch_dtype=torch.bfloat16).to('cuda')
    
    # Test 1: Original prompt with 96 tokens
    prompts_orig = [t['prompt'] for t in templates]
    enc_orig = tok(prompts_orig, return_tensors='pt', padding=True).to('cuda')
    with torch.no_grad():
        out_96 = m.generate(**enc_orig, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id)
    dec_96 = tok.batch_decode(out_96[:, enc_orig['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print("\n--- Original prompt @ 96 tokens ---")
    for t, g in zip(templates, dec_96):
        p, r = eval_code_submission(g, t['template_name'])
        status = "PASS" if p else "FAIL"
        print(f"  {t['template_name']:<15}: {status:<6} | {r}")
        if not p:
            print(f"     RAW: {repr(g.strip()[:100])}")
    p_96 = sum(1 for t, g in zip(templates, dec_96) if eval_code_submission(g, t['template_name'])[0])
    print(f"Original prompt @ 96 tokens: {p_96}/10 ({p_96*10}%)")

    # Test 2: In-fill prompt: prompt includes 'def func_name(...):'
    func_headers = {
        "square": "def square(x):\n",
        "is_even": "def is_even(x):\n",
        "get_length": "def get_length(lst):\n",
        "reverse_string": "def reverse_string(s):\n",
        "find_max": "def find_max(a, b):\n",
        "double_nums": "def double_nums(nums):\n",
        "first_elem": "def first_elem(lst):\n",
        "is_positive": "def is_positive(x):\n",
        "cube": "def cube(n):\n",
        "concat": "def concat(a, b):\n",
    }
    prompts_infill = [t['prompt'] + func_headers[t['template_name']] for t in templates]
    enc_infill = tok(prompts_infill, return_tensors='pt', padding=True).to('cuda')
    with torch.no_grad():
        out_infill = m.generate(**enc_infill, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    dec_infill = tok.batch_decode(out_infill[:, enc_infill['input_ids'].shape[1]:], skip_special_tokens=True)

    print("\n--- Infill prompt (signature provided) @ 64 tokens ---")
    for t, g in zip(templates, dec_infill):
        # We prepend the header since the model continues from it
        full_code = func_headers[t['template_name']] + g
        p, r = eval_code_submission(full_code, t['template_name'])
        status = "PASS" if p else "FAIL"
        print(f"  {t['template_name']:<15}: {status:<6} | {r}")
        if not p:
            print(f"     RAW: {repr(full_code.strip()[:100])}")
    p_infill = sum(1 for t, g in zip(templates, dec_infill) if eval_code_submission(func_headers[t['template_name']] + g, t['template_name'])[0])
    print(f"Infill prompt @ 64 tokens: {p_infill}/10 ({p_infill*10}%)")

if __name__ == "__main__":
    main()
