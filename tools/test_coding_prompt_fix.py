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
    
    # Test 1: Original prompt with 64 tokens
    prompts_orig = [t['prompt'] for t in templates]
    enc_orig = tok(prompts_orig, return_tensors='pt', padding=True).to('cuda')
    with torch.no_grad():
        out_64 = m.generate(**enc_orig, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    dec_64 = tok.batch_decode(out_64[:, enc_orig['input_ids'].shape[1]:], skip_special_tokens=True)
    
    p_64 = sum(1 for t, g in zip(templates, dec_64) if eval_code_submission(g, t['template_name'])[0])
    print(f"Original prompt @ 64 tokens: {p_64}/10 ({p_64*10}%)")

    # Test 2: Prompt with 'without docstring or comments' @ 64 tokens
    prompts_nodoc = [
        t['prompt'].replace(":\n```python\n", " without docstrings or comments:\n```python\n")
        for t in templates
    ]
    enc_nodoc = tok(prompts_nodoc, return_tensors='pt', padding=True).to('cuda')
    with torch.no_grad():
        out_nodoc = m.generate(**enc_nodoc, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    dec_nodoc = tok.batch_decode(out_nodoc[:, enc_nodoc['input_ids'].shape[1]:], skip_special_tokens=True)

    print("\n--- Results with 'without docstrings' prompt @ 64 tokens ---")
    for t, g in zip(templates, dec_nodoc):
        p, r = eval_code_submission(g, t['template_name'])
        status = "PASS" if p else "FAIL"
        print(f"  {t['template_name']:<15}: {status:<6} | {repr(g.strip()[:60])} | {r}")
    p_nodoc = sum(1 for t, g in zip(templates, dec_nodoc) if eval_code_submission(g, t['template_name'])[0])
    print(f"\nNo-doc prompt @ 64 tokens: {p_nodoc}/10 ({p_nodoc*10}%)")

if __name__ == "__main__":
    main()
